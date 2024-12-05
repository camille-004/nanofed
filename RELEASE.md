# Release Process

This document describes the steps to create a new release of NanoFed.

## Pre-Release Checks

1. **Make sure your local repository is up-to-date**
    ```bash
    git checkout main
    git pull origin main
    ```

2. **Run the automated release preparation script**

    ```bash
    make release-prepare version=<new-version>  # e.g., make release-prepare version=0.1.3
    ```
    This script will:
    - Validate the version format
    - Run all tests
    - Check types
    - Run linting
    - Verify package builds
    - Check for uncommitted changes

## Release Steps

1. **Generate Changelog and Release Notes**

    ```bash
    python scripts/changelog.py v<new-version>  # e.g., python scripts/changelog.py v0.1.3
    ```

2. **Review and Edit Release Notes**
    - Edit `docs/source/release_notes/v<new-version>.rst`
    - Review/edit the generated `CHANGELOG.md`
    - Add any additional notes, breaking changes, or important updates

3. **Commit Changes**

    ```bash
    git add pyproject.toml CHANGELOG.md docs/source/release_notes
    git commit -m "chore: prepare release v<new-version>"
    ```

4. **Create and Push Tag**

    ```bash
    git tag -a v<new-version> -m "Release v<new-version>"
    git push origin main
    git push origin v<new-version>
    ```

## Troubleshooting

### Version Mismatch

If you get a version mismatch error in GitHub Actions:

```bash
git push origin --delete v<version>  # Delete remote tag
git tag -d <version>  # Delete local tag
# Then repeats 3-4 from Release Steps
```

### PyPi Upload Issues

- Versions can never be reused on PyPi, even if deleted
- Always increment to a new version number
- For pre-release versions, use format like 0.1.0a1
