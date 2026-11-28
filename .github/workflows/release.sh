#!/bin/bash
#
# The MIT License (MIT)
#
# Copyright (c) 2026 Kris Jusiak <kris@jusiak.net>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

set -euo pipefail

[ $# -eq 1 ] || { echo "usage: $0 X.Y.Z" >&2; exit 1; }

output=$(
    python -m ruff check src tests bin
    python -m ruff format --check src tests bin
    python -m pytest --cov -q
    python -m coverage report
    python -m coverage json --fail-under=90
    lit -v -j2 --timeout 600 tests/x86_64
    find studies -name '*.ipynb' -not -path '*/.ipynb_checkpoints/*' -print0 | xargs -0 -I{} papermill {} /dev/null
    "$(dirname "${BASH_SOURCE[0]}")/perf.sh"
)
prev_tag=$(git describe --tags --abbrev=0)
changelog=$(git log "$prev_tag"..HEAD --pretty=format:'- %s (%h)')

sed -i -E "s/^version = \"[^\"]*\"/version = \"$1\"/" pyproject.toml
git commit -m ":arrow_up: v$1" -m "$output" pyproject.toml
git tag -a "v$1"
git push origin HEAD:main "v$1"
printf '%s\n\n%s\n' "$changelog" "$output" | gh release create "v$1" --title "v$1" --notes-file -
