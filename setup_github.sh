#!/bin/bash
# setup_github.sh
# ----------------
# Run this once to initialize git and push to GitHub.
# Usage: bash setup_github.sh YoanaKC

set -e

USERNAME=${1:-"YoanaKC"}
REPO="vitaiq"

echo "=== VitaIQ GitHub Setup ==="
echo "Username: $USERNAME"
echo "Repo:     $REPO"
echo ""

# Initialize git
git init
git add .
git commit -m "Initial commit: VitaIQ RAG wellness assistant — ITAI 2377 Group 1"

# Create repo on GitHub (requires gh CLI)
if command -v gh &> /dev/null; then
  echo "Creating GitHub repo via gh CLI..."
  gh repo create "$REPO" --public --description "VitaIQ: RAG-based wellness & longevity AI assistant — ITAI 2377 HCC" --source=. --push
  echo "Done! Visit: https://github.com/$USERNAME/$REPO"
else
  echo ""
  echo "gh CLI not found. Manual steps:"
  echo "  1. Go to https://github.com/new"
  echo "  2. Create a repo named '$REPO' (public)"
  echo "  3. Then run:"
  echo "     git remote add origin https://github.com/$USERNAME/$REPO.git"
  echo "     git branch -M main"
  echo "     git push -u origin main"
fi
