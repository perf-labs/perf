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

import platform

from . import x86_64

_ARCHES = {
    "x86-64": x86_64,
    "x86_64": x86_64,
    "amd64": x86_64,
}


def arch(name=None):
    if name is None:
        name = platform.machine()
    try:
        inner = getattr(name, "arch", None)
        if inner is not None:
            candidate = getattr(inner, "name", None)
            if isinstance(candidate, str) and candidate:
                name = candidate
            else:
                name = inner
        elif not isinstance(name, str):
            candidate = getattr(name, "name", None)
            if isinstance(candidate, str) and candidate:
                name = candidate
    except Exception:
        pass
    arch = _ARCHES.get(str(name).lower())
    if arch is None:
        raise ValueError(
            f"unsupported architecture {name!r}; supported: "
            f"{', '.join(sorted(_ARCHES))}"
        )
    return arch


def load(project):
    return arch(project)


__all__ = ["arch", "x86_64"]
