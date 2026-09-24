"""Skill installation preserves source and prior copies across failures."""

from pathlib import Path

import pytest

import transect.skills as skills


def write_skill(root: Path, name: str, text: str) -> Path:
    """Create a disposable skill, never touching the packaged source tree."""
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(text)
    return directory


def installation(tmp_path):
    """Return isolated source, project and existing destination paths."""
    source = tmp_path / "source"
    write_skill(source, "alpha", "new")
    project = tmp_path / "project"
    destination = write_skill(project / ".claude/skills", "alpha", "old")
    return source, project, destination


def test_aliases_of_the_source_are_left_untouched(tmp_path, monkeypatch):
    """A self-install on an editable checkout and a destination symlinked
    to the source are both already installed: nothing moves."""
    checkout = tmp_path / "checkout"
    source = checkout / ".claude/skills"
    skill = write_skill(source, "alpha", "source")
    original_inode = skill.stat().st_ino
    monkeypatch.setattr(skills, "skill_source", lambda: source)
    monkeypatch.chdir(checkout)
    assert skills.install() == ["alpha"]
    assert skill.stat().st_ino == original_inode
    linked = tmp_path / "project/.claude/skills/alpha"
    linked.parent.mkdir(parents=True)
    linked.symlink_to(skill, target_is_directory=True)
    assert skills.install(tmp_path / "project", source=source) == ["alpha"]
    assert linked.is_symlink()
    assert (skill / "SKILL.md").read_text() == "source"


def test_replacement_removes_stale_files_and_preserves_unrelated_skills(tmp_path):
    """Successful replacement refreshes one skill without changing neighbors."""
    source, project, destination = installation(tmp_path)
    (destination / "stale.md").write_text("stale")
    other = write_skill(destination.parent, "unrelated", "other")
    assert skills.install(project, source=source) == ["alpha"]
    assert (destination / "SKILL.md").read_text() == "new"
    assert not (destination / "stale.md").exists()
    assert (other / "SKILL.md").read_text() == "other"


@pytest.mark.parametrize("step", ["copy", "swap"])
def test_any_replacement_failure_preserves_the_prior_copy(tmp_path, monkeypatch, step):
    """Whether the staged copy or the final swap fails, the existing
    skill survives in place and no staging debris is left behind."""
    source, project, destination = installation(tmp_path)
    if step == "copy":

        def fail_copy(src, dest):
            Path(dest).mkdir(parents=True)
            raise OSError("copy failed")

        monkeypatch.setattr(skills.shutil, "copytree", fail_copy)
    else:
        replace = Path.replace
        state = {"failed": False}

        def fail_swap(path, target):
            if Path(target) == destination and not state["failed"]:
                state["failed"] = True
                raise OSError("swap failed")
            return replace(path, target)

        monkeypatch.setattr(Path, "replace", fail_swap)
    with pytest.raises(OSError, match="failed"):
        skills.install(project, source=source)
    assert (destination / "SKILL.md").read_text() == "old"
    assert list(destination.parent.iterdir()) == [destination]


@pytest.mark.parametrize("recovery", ["error", "interrupt"])
def test_failed_recovery_retains_the_old_copy_as_backup(
    tmp_path, monkeypatch, recovery
):
    """When restoring the backup fails too (error or interrupt), the old
    directory survives on disk instead of being cleaned away."""
    source, project, destination = installation(tmp_path)
    replace = Path.replace
    attempts = {"n": 0}

    def fail_destination(path, target):
        if Path(target) == destination:
            attempts["n"] += 1
            if attempts["n"] > 1 and recovery == "interrupt":
                raise KeyboardInterrupt()
            raise OSError("destination unavailable")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_destination)
    expected = KeyboardInterrupt if recovery == "interrupt" else OSError
    with pytest.raises(expected) as raised:
        skills.install(project, source=source)
    survivors = [
        p for p in destination.parent.rglob("SKILL.md") if p.read_text() == "old"
    ]
    assert len(survivors) == 1
    if recovery == "error":
        assert str(survivors[0].parent) in str(raised.value)


@pytest.mark.parametrize("broken", [False, True])
def test_destination_symlink_replacement_preserves_target(tmp_path, broken):
    """Replacing a destination link never deletes or mutates its target."""
    source = tmp_path / "source"
    write_skill(source, "alpha", "new")
    project = tmp_path / "project"
    destination = project / ".claude/skills/alpha"
    destination.parent.mkdir(parents=True)
    target = tmp_path / "target"
    if not broken:
        target.mkdir()
        (target / "SKILL.md").write_text("target")
    destination.symlink_to(target, target_is_directory=True)
    assert skills.install(project, source=source) == ["alpha"]
    assert not destination.is_symlink()
    assert (destination / "SKILL.md").read_text() == "new"
    if broken:
        assert not target.exists()
    else:
        assert (target / "SKILL.md").read_text() == "target"


@pytest.mark.parametrize("nested", ["source_in_destination", "destination_in_source"])
def test_overlapping_source_and_destination_are_rejected_before_mutation(
    tmp_path, nested
):
    """Either nesting direction is refused up front; nothing is copied
    or deleted."""
    if nested == "source_in_destination":
        project = tmp_path / "project"
        source = project / ".claude/skills/alpha/nested/source"
    else:
        source = tmp_path / "source"
        project = source / "alpha/nested/project"
    skill = write_skill(source, "alpha", "source")
    with pytest.raises(ValueError, match="overlap"):
        skills.install(project, source=source)
    assert (skill / "SKILL.md").read_text() == "source"
    if nested == "destination_in_source":
        assert not project.exists()
