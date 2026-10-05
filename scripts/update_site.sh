#!/bin/bash
# Usage: bash scripts/update_site.sh
cd "$(dirname "$0")/.."
GROQ_API_KEY=$(grep GROQ ~/.bashrc | cut -d'"' -f2 | tr -d '\n') python3 scripts/generate_site.py
git add docs/index.html
git commit -m "update results site"
git push
