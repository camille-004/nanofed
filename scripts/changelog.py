import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


class Changelog:
    def __init__(self, repo_path: str = ".") -> None:
        self.repo_path = Path(repo_path)
        self.changelog_path = self.repo_path / "CHANGELOG.md"
        self.release_notes_dir = (
            self.repo_path / "docs" / "source" / "release_notes"
        )
        self.github_url = self._get_github_url()

        try:
            subprocess.check_output(["git", "rev-parse", "--git-dir"])
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "Not a git repository. Please run from project root."
            ) from e

    def _get_github_url(self) -> str:
        """Get the GitHub repository URL."""
        try:
            url = subprocess.check_output(
                ["git", "config", "--get", "remote.origin.url"],
                universal_newlines=True,
            ).strip()

            if url.startswith("git@github.com:"):
                url = url.replace("git@github.com:", "https://github.com/")

            if url.endswith(".git"):
                url = url[:-4]

            return url
        except subprocess.CalledProcessError:
            return ""

    def get_latest_tag(self) -> str | None:
        """Get the most recent git tag."""
        try:
            return subprocess.check_output(
                ["git", "describe", "--tags", "--abbrev=0"],
                universal_newlines=True,
            ).strip()
        except subprocess.CalledProcessError:
            return None

    def get_commits_since_tag(self, tag: str | None) -> list[str]:
        git_log_cmd = ["git", "log", "--pretty=format:%s (%h)"]
        if tag:
            git_log_cmd.append(f"{tag}..HEAD")

        return subprocess.check_output(
            git_log_cmd, universal_newlines=True
        ).split("\n")

    def parse_conventional_commit(self, commit: str) -> dict[str, str]:
        pattern = (
            r"^(?P<type>\w+)(?:\((?P<scope>[\w-]+)\))?: "
            r"(?P<description>.+) \((?P<hash>\w+)\)$"
        )
        match = re.match(pattern, commit)

        if not match:
            return {
                "type": "other",
                "scope": "",
                "description": commit,
                "hash": "",
            }

        return match.groupdict()

    def categorize_commits(
        self, commits: list[str]
    ) -> dict[str, list[dict[str, str]]]:
        categories = {
            "feat": [],  # New features
            "fix": [],  # Bug fixes
            "docs": [],  # Documentation changes
            "style": [],  # Code style changes
            "refactor": [],  # Code refactoring
            "perf": [],  # Performance improvements
            "test": [],  # Test changes
            "build": [],  # Build system changes
            "ci": [],  # CI changes
            "chore": [],  # Maintenance
            "other": [],  # Uncategorized
        }

        for commit in commits:
            parsed = self.parse_conventional_commit(commit)
            commit_type = parsed["type"]
            if commit_type in categories:
                categories[commit_type].append(parsed)
            else:
                categories["other"].append(parsed)

        return categories

    def generate_rst_changelog(
        self, version: str, categories: dict[str, list[dict[str, str]]]
    ) -> str:
        changelog_items = []

        # Flatten all commits across categories into a single list
        all_commits = [
            commit for commits in categories.values() for commit in commits
        ]

        for commit in all_commits:
            scope = f"({commit['scope']}) " if commit["scope"] else ""
            commit_hash = commit["hash"]
            if self.github_url and commit_hash:
                changelog_items.append(
                    f"* {scope}{commit['description']} "
                    f"`{commit_hash} <{self.github_url}/commit/{commit_hash}>`_"  # noqa
                )
            else:
                changelog_items.append(
                    f"* {scope}{commit['description']} ({commit_hash})"
                )

        return "\n".join(changelog_items)

    def generate_md_changelog(
        self, version: str, categories: dict[str, list[dict[str, str]]]
    ) -> str:
        content = [
            f"## [{version}]({self.github_url}/releases/tag/{version}) - "
            f"{datetime.now(timezone.utc).strftime("%Y-%m-%d")}\n"
        ]

        type_headers = {
            "feat": "Features",
            "fix": "Bug Fixes",
            "docs": "Documentation",
            "style": "Style",
            "refactor": "Code Refactoring",
            "perf": "Performance Improvements",
            "tests": "Tests",
            "build": "Build System",
            "ci": "Continuous Integration",
            "chore": "Chores",
            "other": "Other Changes",
        }

        for category, commits in categories.items():
            if commits:
                content.append(f"### {type_headers[category]}\n")
                for commit in commits:
                    scope = f"({commit['scope']}) " if commit["scope"] else ""
                    commit_hash = commit["hash"]
                    if self.github_url and commit_hash:
                        commit_url = f"{self.github_url}/commit/{commit_hash}"
                        content.append(
                            f"- {scope}{commit['description']} ([{commit_hash}]({commit_url}))"  # noqa
                        )
                    else:
                        content.append(
                            f"- {scope}{commit['description']} ({commit_hash})"
                        )
                content.append("")

        return "\n".join(content)

    def update_changelog(self, version: str) -> None:
        latest_tag = self.get_latest_tag()
        commits = self.get_commits_since_tag(latest_tag)

        if not commits or commits[0] == "":
            print("No commits found since last tag.")
            return

        categories = self.categorize_commits(commits)

        # Update the Markdown changelog
        md_content = self.generate_md_changelog(version, categories)
        md_updated = False

        if self.changelog_path.exists():
            current_content = self.changelog_path.read_text()

            version_header = f"## [{version}]"
            if version_header not in current_content:
                if "# Changelog" in current_content:
                    header_pos = current_content.find("# Changelog")
                    updated = (
                        current_content[:header_pos]
                        + "# Changelog\n\n"
                        + md_content
                        + "\n"
                        + current_content[header_pos + 10 :].lstrip()
                    )
                else:
                    updated = f"# Changelog\n\n{md_content}\n"
                self.changelog_path.write_text(updated)
                md_updated = True
        else:
            updated = f"# Changelog\n\n{md_content}\n"
            self.changelog_path.write_text(updated)
            md_updated = True

        if md_updated:
            print(
                f"Markdown changelog updated for version {version} "
                f"at {self.changelog_path}"
            )
        else:
            print(
                f"Markdown changelog already up-to-date for version {version}"
            )

        rst_path = self.release_notes_dir / f"v{version.lstrip('v')}.rst"
        if not rst_path.exists():
            print(f"Warning: Release notes file not found at {rst_path}")
            print("Please run make release-prepare first.")
            return

        rst_content = rst_path.read_text()
        marker = ".. Generated automatically from git commits"
        generated_changelog = self.generate_rst_changelog(version, categories)

        if marker in rst_content:
            before_changelog = rst_content.split(marker)[0]
            updated_content = (
                f"{before_changelog}{marker}\n\n{generated_changelog}\n"
            )
            rst_path.write_text(updated_content)
            print(f"RST changelog updated for version {version} at {rst_path}")
        else:
            updated_content = (
                f"{rst_content}\n\n{marker}\n\n{generated_changelog}\n"
            )
            rst_path.write_text(updated_content)
            print(
                f"RST changelog appended for version {version} at {rst_path}"
            )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate changelog from git commits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/changelog.py v1.0.1

Note: run this script from the project root directory.
Make sure your commits follow for the Conventional Commits format:
    <type>(<scope>): <description>

    Types: feat, fix, docs, style, refactor, perf, test, build, ci, chore
    Example: feat(api): add new endpoint
    """,
    )

    parser.add_argument("version", help="New version number (e.g., v1.0.1)")
    parser.add_argument(
        "--path",
        default=".",
        help="Path to repository root (default: current directory)",
    )

    args = parser.parse_args()

    try:
        generator = Changelog(args.path)
        generator.update_changelog(args.version)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
