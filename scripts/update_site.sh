#!/bin/bash
# Usage: bash scripts/update_site.sh
cd "$(dirname "$0")/.."
source ~/.bashrc
python3 scripts/generate_site.py
git add docs/index.html
git commit -m "update results site"
git push
