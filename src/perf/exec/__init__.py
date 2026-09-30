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

from .elf import (
    Elf,
    ElfConst,
    align_down,
    align_up,
    call_native,
    initializers,
    is_archive,
    is_relocatable,
    is_shared_library,
    link_object,
    load_shared_libraries,
    needs_link,
    perf_map_path,
    resolve_exec,
    retire_exit_hooks,
    shared_libraries,
    to_object,
    write_perf_map,
)

__all__ = [
    "Elf",
    "ElfConst",
    "align_down",
    "align_up",
    "call_native",
    "initializers",
    "is_archive",
    "is_relocatable",
    "is_shared_library",
    "link_object",
    "load_shared_libraries",
    "needs_link",
    "perf_map_path",
    "resolve_exec",
    "retire_exit_hooks",
    "shared_libraries",
    "to_object",
    "write_perf_map",
]
