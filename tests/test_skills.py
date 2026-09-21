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


def test_default_editable_self_install_preserves_source(tmp_path, monkeypatch):
    """Default installation in an editable checkout is an unchanged no-op."""
    project = tmp_path / "checkout"
    source = project / ".claude/skills"
    skill = write_skill(source, "alpha", "source")
    original_inode = skill.stat().st_ino
    monkeypatch.setattr(skills, "skill_source", lambda: source)
    monkeypatch.chdir(project)
    assert skills.install() == ["alpha"]
    assert (skill / "SKILL.md").read_text() == "source"
    assert skill.stat().st_ino == original_inode
    alias = tmp_path / "alias"
    alias.symlink_to(project, target_is_directory=True)
    assert skills.install(alias, source=source) == ["alpha"]
    assert (skill / "SKILL.md").read_text() == "source"
    assert skill.stat().st_ino == original_inode
    assert alias.is_symlink()


def test_replacement_removes_stale_files_and_preserves_unrelated_skills(tmp_path):
    """Successful replacement refreshes one skill without changing neighbors."""
    source, project, destination = installation(tmp_path)
    (destination / "stale.md").write_text("stale")
    other = write_skill(destination.parent, "unrelated", "other")
    assert skills.install(project, source=source) == ["alpha"]
    assert (destination / "SKILL.md").read_text() == "new"
    assert not (destination / "stale.md").exists()
    assert (other / "SKILL.md").read_text() == "other"
    assert sorted(path.name for path in destination.parent.iterdir()) == [
        "alpha",
        "unrelated",
    ]


def test_copy_failure_preserves_prior_contents(tmp_path, monkeypatch):
    """A partially copied replacement never removes the existing skill."""
    source, project, destination = installation(tmp_path)

    def fail_copy(src, dest):
        Path(dest).mkdir(parents=True)
        (Path(dest) / "partial").write_text("incomplete")
        raise OSError("copy failed")

    monkeypatch.setattr(skills.shutil, "copytree", fail_copy)
    with pytest.raises(OSError, match="copy failed"):
        skills.install(project, source=source)
    assert (destination / "SKILL.md").read_text() == "old"
    assert (source / "alpha/SKILL.md").read_text() == "new"
    assert list(destination.parent.iterdir()) == [destination]


def test_replacement_failure_restores_previous_copy(tmp_path, monkeypatch):
    """A failed final rename rolls the saved prior directory back into place."""
    source, project, destination = installation(tmp_path)
    replace = Path.replace
    failed = False

    def fail_once(path, target):
        nonlocal failed
        if Path(target) == destination and not failed:
            failed = True
            raise OSError("replacement failed")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_once)
    with pytest.raises(OSError, match="replacement failed"):
        skills.install(project, source=source)
    assert (destination / "SKILL.md").read_text() == "old"
    assert list(destination.parent.iterdir()) == [destination]


def test_failed_rollback_retains_recoverable_backup(tmp_path, monkeypatch):
    """If rollback fails too, the old directory survives at the reported backup."""
    source, project, destination = installation(tmp_path)
    replace = Path.replace

    def fail_destination(path, target):
        if Path(target) == destination:
            raise OSError("destination unavailable")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_destination)
    with pytest.raises(OSError, match="backup") as raised:
        skills.install(project, source=source)
    backups = [
        path.parent
        for path in destination.parent.rglob("SKILL.md")
        if path.read_text() == "old"
    ]
    assert len(backups) == 1 and str(backups[0]) in str(raised.value)
    assert (source / "alpha/SKILL.md").read_text() == "new"


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


def test_destination_link_to_source_is_unchanged(tmp_path):
    """A destination alias of the skill source is already installed."""
    source = tmp_path / "source"
    skill = write_skill(source, "alpha", "source")
    project = tmp_path / "project"
    destination = project / ".claude/skills/alpha"
    destination.parent.mkdir(parents=True)
    destination.symlink_to(skill, target_is_directory=True)
    assert skills.install(project, source=source) == ["alpha"]
    assert destination.is_symlink()
    assert (skill / "SKILL.md").read_text() == "source"


def test_overlapping_destination_cannot_delete_nested_source(tmp_path):
    """A destination containing the source is rejected before mutation."""
    project = tmp_path / "project"
    destination = project / ".claude/skills/alpha"
    source = destination / "nested/source"
    skill = write_skill(source, "alpha", "source")
    with pytest.raises(ValueError, match="overlap"):
        skills.install(project, source=source)
    assert (skill / "SKILL.md").read_text() == "source"


def test_interrupted_rollback_retains_old_copy(tmp_path, monkeypatch):
    """An interruption during recovery cannot clean up the sole prior copy."""
    source, project, destination = installation(tmp_path)
    replace = Path.replace
    attempts = 0

    def interrupt_recovery(path, target):
        nonlocal attempts
        if Path(target) == destination:
            attempts += 1
            if attempts == 1:
                raise OSError("replacement failed")
            raise KeyboardInterrupt()
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", interrupt_recovery)
    with pytest.raises(KeyboardInterrupt):
        skills.install(project, source=source)
    assert any(
        path.read_text() == "old" for path in destination.parent.rglob("SKILL.md")
    )


def test_source_containing_destination_is_rejected_before_copy(tmp_path):
    """Installing inside the source skill cannot recursively copy itself."""
    source = tmp_path / "source"
    skill = write_skill(source, "alpha", "source")
    project = skill / "nested/project"
    with pytest.raises(ValueError, match="overlap"):
        skills.install(project, source=source)
    assert (skill / "SKILL.md").read_text() == "source"
    assert not project.exists()
