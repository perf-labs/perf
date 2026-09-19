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

from . import plot as _plot
from .arch import arch, load
from .bench import bench, disassemble, to_json
from .compare import compare
from .data import is_record, metrics, parse, samples
from .exec import to_object
from .info import (
    bin,
    cpuinfo,
    functions,
    labels,
    metadata,
    regions,
    targets,
)
from .track import (
    track,
)

plot = _plot.PlotConfig()

__all__ = [
    "arch",
    "load",
    "bench",
    "compare",
    "disassemble",
    "to_json",
    "parse",
    "metrics",
    "samples",
    "is_record",
    "to_object",
    "bin",
    "cpuinfo",
    "metadata",
    "labels",
    "functions",
    "regions",
    "targets",
    "track",
    "plot",
]
