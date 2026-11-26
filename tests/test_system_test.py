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

import importlib.machinery
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd


def _system_test():
    path = (
        Path(__file__).resolve().parent.parent / "bin" / "x86_64" / "perf-system-test"
    )
    loader = importlib.machinery.SourceFileLoader("perf_system_test", str(path))
    spec = importlib.util.spec_from_loader("perf_system_test", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class TestSystemTestExit(unittest.TestCase):
    def test_mismatch_fails_and_exits_non_zero(self):
        module = _system_test()
        measured = {"clock": 1.0, "documented_clock": 100.0}
        with patch.object(module, "_st_measure", return_value=measured):
            frame = module._st_test(iterations=1, samples=1)
        self.assertFalse(frame.attrs["ok"])
        clock = frame.loc[frame["property"] == "Clock"].iloc[0]
        self.assertEqual(clock["status"], "fail")

        failed = pd.DataFrame([{"property": "Clock", "status": "fail"}])
        failed.attrs["ok"] = False
        args = SimpleNamespace(cpu=None, output=None, interactive=False)
        with patch.object(module, "_st_test", return_value=failed):
            with patch.object(module, "show_table", return_value=False):
                with self.assertRaises(SystemExit) as error:
                    module._run(args, {"iterations": 1, "samples": 1})
        self.assertEqual(error.exception.code, 1)

    def test_compare_uses_a_documented_range(self):
        module = _system_test()
        self.assertEqual(module._st_compare(125.0, 100.0)[0], "ok")
        self.assertEqual(module._st_compare(125.1, 100.0)[0], "fail")
        self.assertEqual(module._st_compare(1.9, 1.0)[0], "ok")
        self.assertEqual(module._st_compare(2.1, 1.0)[0], "fail")


if __name__ == "__main__":
    unittest.main()
