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

import logging as _logging
from types import SimpleNamespace as _SimpleNamespace

_logging.getLogger("angr.state_plugins.unicorn_engine").setLevel(_logging.CRITICAL)

from . import plot as _plot  # noqa: E402
from .bench import benchmark  # noqa: E402
from .cmp import compare  # noqa: E402
from .info import cpuinfo as _cpuinfo  # noqa: E402
from .mca import analyze  # noqa: E402
from .opt import optimize  # noqa: E402
from .prof import profile  # noqa: E402
from .tune import (  # noqa: E402
    reset as _reset,
    status as _status,
    test as _test,
    tune as _tune,
)

system = _SimpleNamespace(
    info=_cpuinfo, status=_status, tune=_tune, reset=_reset, test=_test
)
__all__ = [
    "system",
    "benchmark",
    "profile",
    "analyze",
    "compare",
    "optimize",
    "plot",
]

plot = _plot.PlotConfig()
