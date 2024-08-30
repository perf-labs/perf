#
# The MIT License (MIT)
#
# Copyright (c) 2024-2025 Kris Jusiak <kris@jusiak.net>
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
#
# Requirements:
#   - apt-get install coreutils # base64
#   - apt-get install imagemagick # convert
#   - apt-get install libsixel-bin # sixel
#   - pip install ansi2html # ansi2html
#
# Usage:
#   ./app 2>&1 | ./export.sh markdown > app.md
#   ./app 2>&1 | ./export.sh notebook > app.ipynb
#   ./app 2>&1 | ./export.sh html > app.html

function markdown() {
  rm -rf tmp
  mkdir tmp
  stdin=$(mktemp)
  stdout=$(mktemp)

  cat - > $stdin

  i=0
  while grep -q $'\x1bP' $stdin; do
    sed -z 's/\(.*\)\x1bP.*/\1/' $stdin >> $stdout
    sed -z 's/.*\(\x1bP.*\x1b\\\).*/\1/' $stdin | \
      convert sixel:- png:- > tmp/$((i++)).png
    sed -z 's/.*\x1b\\\(.*\)/\1/' $stdin >> $stdout
    mv $stdout $stdin
  done

  cat $stdin > tmp/$((i++))
}

function html() {
  stdin=$(mktemp)
  stdout=$(mktemp)

  cat - | ansi2html > $stdin

  while grep -q $'\x1bP' $stdin; do
    sed -z 's/\(.*\)\x1bP.*/\1/' $stdin >> $stdout
    sed -z 's/.*\(\x1bP.*\x1b\\\).*/\1/' $stdin | \
      convert sixel:- png:- | base64 | tr -d '\n' | \
      xargs -i{} echo "<img src='data:image/png;base64,{}'/><br />" >> $stdout
    sed -z 's/.*\x1b\\\(.*\)/\1/' $stdin >> $stdout
    mv $stdout $stdin
  done
  cat $stdin
}

function notebook() {
  cat <<EOF
  {
   "cells": [
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
       $(cat - | markdown |  sed 's/.*/"&\\n",/')
       ""
     ]
    }
   ],
   "metadata": {
    "jupytext": {
     "cell_metadata_filter": "-all",
     "main_language": "python",
     "notebook_metadata_filter": "-all"
    }
   },
   "nbformat": 4,
   "nbformat_minor": 5
  }
EOF
}

if [ $# -lt 1 ]; then
  echo "Usage: $0 <markdown|notebook|html> # reads from stdin"
  exit 1
fi

cat - | $1
