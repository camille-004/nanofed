import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import tomlkit


def get_current_version() -> str:
    """Get current version from pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    with open(pyproject_path) as f:
        pyproject = tomlkit.load(f)
    return pyproject["tool"]["poetry"]["version"]


def validate_version(version: str) -> bool:
    """Validate semantic version format."""
    pattern = r"^\d+\.\d+\.\d+(?:-(?:alpha|beta|rc)\.\d+)?$"
    return bool(re.match(pattern, version))


def create_release_notes(version: str) -> None:
    """Create release notes file for the new version from template."""
    release_notes_dir = Path("docs/source/release_notes")
    template_path = release_notes_dir / "template.rst"
    release_notes_path = release_notes_dir / f"v{version}.rst"

    if release_notes_path.exists():
        return

    if not template_path.exists():
        print(f"❌ Template file not found at {template_path}")
        return

    today = datetime.now(timezone.utc).strftime("%B %d, %Y")

    # Read and update template content
    template_content = template_path.read_text()
    release_content = template_content.replace("{version}", version).replace(
        "{release_date}", today
    )

    # Write the new release notes file
    release_notes_path.write_text(release_content)
    print(f"✅ Created release notes at {release_notes_path}")


def run_checks() -> tuple[bool, list[str]]:
    """Run pre-release checks."""
    checks = []
    success = True

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        if result.stdout:
            checks.append("❌ You have uncommitted changes")
            success = False
        else:
            checks.append("✅ Git working directory is clean")

        # Run tests
        subprocess.run(["poetry", "run", "pytest"], check=True)
        checks.append("✅ All tests passed")

        # Run type checks
        subprocess.run(["poetry", "run", "mypy", "nanofed"], check=True)
        checks.append("✅ Type checking passed")

        # Run linting
        subprocess.run(
            [
                "poetry",
                "run",
                "ruff",
                "check",
                "nanofed/",
                "scripts/",
                "tests/",
            ],
            check=True,
        )
        checks.append("✅ Linting passed")

        subprocess.run(["poetry", "build"], check=True)
        checks.append("✅ Package builds successfully")

    except subprocess.CalledProcessError as e:
        checks.append(f"❌ Failed: {e.cmd}")
        success = False

    return success, checks


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Prepare a new release")
    parser.add_argument("new_version", help="New version number (e.g., 0.2.0)")
    args = parser.parse_args()

    if not validate_version(args.new_version):
        print(
            "❌ Invalid version format. Use semantic versioning (e.g., 0.2.0)"
        )
        return

    current_version = get_current_version()
    print(f"Current version: {current_version}")
    print(f"Preparing release: {args.new_version}")

    success, checks = run_checks()
    print("\nPre-release checks:")
    for check in checks:
        print(check)

    if not success:
        print("\n❌ Pre-release checks failed. Please fix the issues above.")
        return

    # Update version
    subprocess.run(["poetry", "version", args.new_version], check=True)

    # Create release notes template
    create_release_notes(args.new_version)

    print("\n✅ Ready for release!")
    print("\nNext steps:")
    print("1. Review and update the release notes file")
    print("2. Run `make release` to complete the release process")


if __name__ == "__main__":
    main()
