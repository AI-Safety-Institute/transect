"""Install the packaged Claude Code skills into a project.

The repo's `.claude/skills/` (using-transect, transect-diagnostics,
add-a-layer, custom-ui) ship inside the wheel as `transect/_skills`;
`python -m transect.skills install` copies them
into the current project's `.claude/skills/` so Claude Code discovers
them there. Re-running after a package upgrade refreshes the copies.
"""

from __future__ import annotations

import shutil
import sys
from importlib.resources import files
from pathlib import Path


def skill_source() -> Path:
    """The shipped skills directory: the wheel's `transect/_skills`, or the
    repo checkout's `.claude/skills` on an editable/source install.

    Wheels install unpacked, so the packaged traversable is a real
    directory whenever it exists."""
    packaged = Path(str(files("transect").joinpath("_skills")))
    if packaged.is_dir():
        return packaged
    repo = Path(__file__).resolve().parents[2] / ".claude" / "skills"
    if repo.is_dir():
        return repo
    raise FileNotFoundError(
        "no shipped skills found: this install carries no transect/_skills "
        "and no repo .claude/skills sits alongside the package"
    )


def install(project_dir: str | Path = ".", source: Path | None = None) -> list[str]:
    """Copy every shipped skill into ``project_dir``/.claude/skills/,
    replacing any prior copy of the same skill (the skills version
    with the package). Other skills in the project are untouched.

    Returns the installed skill names, sorted.
    """
    src = skill_source() if source is None else source
    skills = sorted(p for p in src.iterdir() if p.is_dir())
    if not skills:
        raise FileNotFoundError(f"{src} contains no skill directories")
    dest_root = Path(project_dir) / ".claude" / "skills"
    for skill in skills:
        dest = dest_root / skill.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(skill, dest)
    return [p.name for p in skills]


def main(argv: list[str]) -> int:
    if not argv or argv[0] != "install" or len(argv) > 2:
        print("usage: python -m transect.skills install [project_dir]", file=sys.stderr)
        return 2
    project_dir = Path(argv[1]) if len(argv) == 2 else Path.cwd()
    names = install(project_dir)
    dest = project_dir / ".claude" / "skills"
    print(f"installed {', '.join(names)} -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
