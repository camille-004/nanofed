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
    - Update the library version in `pyproject.toml`
    - Run all tests
    - Check types
    - Run linting
    - Verify package builds
    - Check for uncommitted changes

## Release Steps

The release process is automated via a single command after the pre-release checks:

```bash
make release
```

This command will:

1. Validate that you're on the main branch with a clean working directory
2. Generate the changelog automatically
3. Pause for you to review and edit the release notes:
    - Edit `docs/source/release_notes/v<new-version>.rst`
    - Add any additional notes, breaking changes, or important updates
4. Commit all changes
6. Create and push the version tag

### Next Steps

After running `make release`, you should:

1. Wait for CI to compelte
2. Monitor PyPI release
3. Verify documentation update

## Troubleshooting

### PyPi Upload Issues

- Versions can never be reused on PyPi, even if deleted
- Always increment to a new version number
- For pre-release versions, use format like 0.1.0a1
