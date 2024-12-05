import re
import subprocess
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
    print("Preparing release: args.new_version")

    success, checks = run_checks()
    print("\nPre-release checks:")
    for check in checks:
        print(check)

    if not success:
        print("\n❌ Pre-release checks failed. Please fix the issues above.")
        return

    # Update version
    subprocess.run(["poetry", "version", args.new_version], check=True)

    print("\n✅ Ready for release!")
    print("\nNext steps:")
    print(
        f"1. Commit the version change: git commit -am 'chore: bump version "
        f"to {args.new_version}'"
    )
    print(
        f"2. Create and push tag: git tag -a v{args.new_version} -m 'Release "
        f"v{args.new_version}' && git push --tags"
    )


if __name__ == "__main__":
    main()
