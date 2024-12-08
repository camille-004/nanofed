#!/bin/bash
set -eou pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

step() {
    echo -e "${GREEN}==>${NC} $1"
}

error() {
    echo -e "${RED}Error: $1${NC}" >&2
    exit 1
}

# Check if command exists
check_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        error "Required command '$1' not found"
    fi
}

validate_requirements() {
    check_command git
    check_command poetry
    check_command python

    current_branch=$(git branch --show-current)
    if [ "$current_branch" != "main" ]; then
        error "Must be on main branch to release (currently on '$current_branch')"
    fi

    step "Pulling latest changes from main..."
    git pull origin main
}

get_version() {
    poetry version --short
}

check_release_notes() {
    local version=$1
    local release_notes="docs/source/release_notes/v${version}.rst"

    if [ ! -f "$release_notes" ]; then
        error "Release notes file not found: $release_notes"
    fi

    local notes_time=$(stat -f %m "$release_notes" 2>/dev/null || stat -c %Y "$release_notes")
    local changelog_time=$(stat -f %m "CHANGELOG.md" 2>/dev/null || stat -c %Y "CHANGELOG.md")

    if [ "$notes_time" -lt "$changelog_time" ]; then
        error "Release notes must be updated after changelog generation"
    fi
}

main() {
    step "Validating requirements..."
    validate_requirements

    VERSION=$(get_version)
    step "Preparing release for version v${VERSION}"

    step "Generating changelog..."
    python scripts/changelog.py "v${VERSION}"

    echo -e "\n${YELLOW}Please review and update the release notes now:${NC}"
    echo " docs/source/release_notes/v${VERSION}.rst"
    echo -e "${YELLOW}Press Enter once you've reviewed the updated release notes...${NC}"
    read -r

    step "Verifying release notes..."
    check_release_notes "$VERSION"

    step "Committing changes..."
    git add .
    git commit -m "chore: prepare release v${VERSION}"

    step "Creating and pushing tag..."
    git tag -a "v${VERSION}" -m "Release v${VERSION}"
    git push origin main
    git push origin "v${VERSION}"

    echo -e "\n${GREEN}Release v${VERSION} has been prepared and pushed!${NC}"
    echo -e "\nNext steps:"
    echo "1. Wait for CI to complete"
    echo "2. Monitor PyPI release"
    echo "3. Verify documentation update"
}

main "$@"
