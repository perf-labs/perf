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

import errno
import glob
import json
import re
from pathlib import Path

import pandas as pd

_COLUMNS = ("setting", "path", "current", "target", "status")
_DEFAULT_PROFILE = "default"
_FREQ_MIN = "/sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq"
_FREQ_MAX = "/sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq"
_TUNABLE = (
    "aslr",
    "governor",
    "energy_performance_preference",
    "cpu_frequency",
    "turbo",
    "boost",
    "smt",
    "nmi_watchdog",
    "numa_balancing",
    "thp",
    "thp_defrag",
    "khugepaged_defrag",
    "ksm",
    "swappiness",
    "stat_interval",
    "timer_migration",
    "perf_event_max_sample_rate",
    "perf_event_mlock_kb",
    "perf_event_paranoid",
)
_SETTINGS = {
    "aslr": {
        "path": "/proc/sys/kernel/randomize_va_space",
        "kind": "aslr",
    },
    "governor": {
        "path": "/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor",
        "kind": "text",
        "available": "scaling_available_governors",
    },
    "energy_performance_preference": {
        "path": "/sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference",
        "kind": "text",
        "available": "energy_performance_available_preferences",
    },
    "cpu_frequency": {
        "path": "/sys/devices/system/cpu/cpu*/cpufreq/scaling_{min,max}_freq",
        "kind": "freq",
    },
    "turbo": {
        "path": "/sys/devices/system/cpu/intel_pstate/no_turbo",
        "kind": "inverted_bool",
    },
    "boost": {
        "path": "/sys/devices/system/cpu/cpufreq/boost",
        "kind": "bool",
    },
    "smt": {
        "path": "/sys/devices/system/cpu/smt/control",
        "kind": "smt",
    },
    "nmi_watchdog": {
        "path": "/proc/sys/kernel/nmi_watchdog",
        "kind": "bool",
    },
    "numa_balancing": {
        "path": "/proc/sys/kernel/numa_balancing",
        "kind": "bool",
    },
    "thp": {
        "path": "/sys/kernel/mm/transparent_hugepage/enabled",
        "kind": "choice",
        "choices": ("always", "madvise", "never"),
    },
    "thp_defrag": {
        "path": "/sys/kernel/mm/transparent_hugepage/defrag",
        "kind": "choice",
        "choices": ("always", "madvise", "defer", "defer+madvise", "never"),
    },
    "khugepaged_defrag": {
        "path": "/sys/kernel/mm/transparent_hugepage/khugepaged/defrag",
        "kind": "bool",
    },
    "ksm": {
        "path": "/sys/kernel/mm/ksm/run",
        "kind": "bool",
    },
    "swappiness": {
        "path": "/proc/sys/vm/swappiness",
        "kind": "int",
        "min": 0,
        "max": 200,
    },
    "stat_interval": {
        "path": "/proc/sys/vm/stat_interval",
        "kind": "int",
        "min": 1,
        "max": 3600,
    },
    "timer_migration": {
        "path": "/proc/sys/kernel/timer_migration",
        "kind": "bool",
    },
    "perf_event_max_sample_rate": {
        "path": "/proc/sys/kernel/perf_event_max_sample_rate",
        "kind": "int",
        "min": 1,
        "max": 1000000000,
    },
    "perf_event_mlock_kb": {
        "path": "/proc/sys/kernel/perf_event_mlock_kb",
        "kind": "int",
        "min": 0,
        "max": 16777216,
    },
    "perf_event_paranoid": {
        "path": "/proc/sys/kernel/perf_event_paranoid",
        "kind": "int",
        "min": -1,
        "max": 99,
    },
}
_PROFILES = {
    "default": {
        "aslr": True,
        "governor": "powersave",
        "energy_performance_preference": "balance_performance",
        "cpu_frequency": "range",
        "turbo": True,
        "boost": True,
        "smt": True,
        "nmi_watchdog": True,
        "numa_balancing": True,
        "thp": "madvise",
        "thp_defrag": "madvise",
        "khugepaged_defrag": True,
        "ksm": False,
        "swappiness": 60,
        "stat_interval": 1,
        "timer_migration": True,
        "perf_event_max_sample_rate": 100000,
        "perf_event_mlock_kb": 516,
        "perf_event_paranoid": 2,
    },
    "latency": {
        "aslr": False,
        "governor": "performance",
        "energy_performance_preference": "performance",
        "cpu_frequency": "max",
        "turbo": True,
        "boost": True,
        "smt": False,
        "nmi_watchdog": False,
        "numa_balancing": False,
        "thp": "never",
        "thp_defrag": "never",
        "khugepaged_defrag": False,
        "ksm": False,
        "swappiness": 0,
        "stat_interval": 120,
        "timer_migration": False,
        "perf_event_max_sample_rate": 1,
        "perf_event_mlock_kb": 65536,
        "perf_event_paranoid": -1,
    },
    "throughput": {
        "aslr": False,
        "governor": "performance",
        "energy_performance_preference": "performance",
        "cpu_frequency": "max",
        "turbo": True,
        "boost": True,
        "smt": True,
        "nmi_watchdog": False,
        "numa_balancing": False,
        "thp": "always",
        "thp_defrag": "always",
        "khugepaged_defrag": True,
        "ksm": False,
        "swappiness": 10,
        "stat_interval": 10,
        "timer_migration": True,
        "perf_event_max_sample_rate": 1,
        "perf_event_mlock_kb": 65536,
        "perf_event_paranoid": -1,
    },
}
_TRUE = ("1", "true", "yes", "on", "enabled", "enable")
_FALSE = ("0", "false", "no", "off", "disabled", "disable")
_THP_CURRENT = re.compile(r"\[([^\]]+)\]")
_PERMISSION_ERRNOS = (errno.EACCES, errno.EPERM)
_FREQ_ALIASES = {
    "auto": "range",
    "default": "range",
    "range": "range",
    "min": "min",
    "low": "min",
    "max": "max",
    "high": "max",
    "base": "base",
    "nominal": "base",
}
_FREQ_KEYS = ("thp", "thp_defrag")


def profiles():
    return tuple(_PROFILES)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        return None
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return None


def _as_int(value):
    if isinstance(value, bool):
        return None
    try:
        return int(str(value).strip(), 0)
    except (TypeError, ValueError):
        return None


def _normalize(name, value):
    spec = _SETTINGS[name]
    kind = spec["kind"]
    if kind in ("bool", "inverted_bool", "smt"):
        on = _as_bool(value)
        if on is None:
            raise ValueError(
                f"profile {name} needs a boolean (true/false), got {value!r}"
            )
        return on
    if kind == "aslr":
        if isinstance(value, bool):
            return value
        number = _as_int(value)
        if number in (0, 1, 2):
            return number
        on = _as_bool(value)
        if on is not None:
            return on
        raise ValueError(
            f"profile aslr needs a boolean (true/false) or 0/1/2, got {value!r}"
        )
    if kind == "freq":
        text = str(value).strip().lower()
        if text in _FREQ_ALIASES:
            return _FREQ_ALIASES[text]
        number = _as_int(value)
        if number is not None and number > 0:
            return number
        raise ValueError(
            f"profile {name} needs min, max, base, range, or a positive "
            f"frequency in kHz, got {value!r}"
        )
    if kind == "choice":
        text = str(value).strip().lower()
        choices = spec.get("choices", ())
        if text not in choices:
            raise ValueError(
                f"profile {name} must be one of "
                f"{', '.join(repr(c) for c in choices)}, got {value!r}"
            )
        return text
    if kind == "int":
        number = _as_int(value)
        low, high = spec.get("min"), spec.get("max")
        if (
            number is None
            or (low is not None and number < low)
            or (high is not None and number > high)
        ):
            raise ValueError(
                f"profile {name} must be an integer in [{low}, {high}], got {value!r}"
            )
        return number
    text = str(value).strip()
    if not text:
        raise ValueError(f"profile {name} must not be empty")
    return text


def _encode(name, value):
    kind = _SETTINGS[name]["kind"]
    if kind == "aslr":
        if isinstance(value, bool):
            return "2" if value else "0"
        return str(value)
    if kind == "bool":
        return "1" if value else "0"
    if kind == "inverted_bool":
        return "0" if value else "1"
    if kind == "smt":
        return "on" if value else "off"
    return str(value)


def _decode(name, raw):
    text = str(raw).strip()
    kind = _SETTINGS[name]["kind"]
    if kind == "aslr":
        return {"0": "off", "1": "partial", "2": "on"}.get(text, text)
    if kind == "bool":
        return "on" if text == "1" else "off" if text == "0" else text
    if kind == "inverted_bool":
        return "on" if text == "0" else "off" if text == "1" else text
    if kind == "smt":
        found = _THP_CURRENT.search(text)
        return found.group(1) if found else text
    if name in _FREQ_KEYS:
        found = _THP_CURRENT.search(text)
        return found.group(1) if found else text
    return text


def _freq_text(value):
    if value is None:
        return ""
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}GHz"
    if value >= 1_000:
        return f"{value / 1_000:g}MHz"
    return f"{value}kHz"


def _display(name, value):
    if isinstance(value, bool):
        return "on" if value else "off"
    if name == "aslr" and isinstance(value, int):
        return {0: "off", 1: "partial", 2: "on"}.get(value, str(value))
    if name == "cpu_frequency" and isinstance(value, int):
        return _freq_text(value)
    return str(value)


def _profile_error(profile):
    return (
        f"unknown profile {profile!r}; expected one of "
        f"{', '.join(_PROFILES)}, or a JSON file (e.g. latency.json)"
    )


def _load_json(profile):
    path = Path(str(profile))
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as ex:
        raise ValueError(f"cannot read profile {profile!r}: {ex}") from ex
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as ex:
        raise ValueError(f"invalid JSON in profile {profile!r}: {ex}") from ex
    if not isinstance(loaded, dict):
        raise ValueError("profile file must contain a JSON object")
    return loaded


def _override_items(overrides):
    if overrides is None:
        return []
    if isinstance(overrides, dict):
        return list(overrides.items())
    if isinstance(overrides, str):
        overrides = [overrides]
    try:
        items = list(overrides)
    except TypeError as ex:
        raise ValueError(f"profile overrides must be a dict, got {overrides!r}") from ex
    out = []
    for item in items:
        if isinstance(item, str):
            key, sep, value = item.partition("=")
            if not sep:
                raise ValueError(
                    f"invalid profile override {item!r}; expected key=value"
                )
            out.append((key, value))
        else:
            try:
                out.extend(item.items())
            except AttributeError as ex:
                raise ValueError(
                    f"invalid profile override {item!r}; expected key=value"
                ) from ex
    return out


def _apply_overrides(target, overrides):
    for key, value in _override_items(overrides):
        name = str(key).strip()
        if name.startswith("profile."):
            name = name[len("profile.") :]
        if name not in _SETTINGS:
            raise ValueError(
                f"unknown profile key {name!r}; expected one of {', '.join(_TUNABLE)}"
            )
        target[name] = _normalize(name, value)
    return target


def _validate_keys(values):
    for key in values:
        if key not in _SETTINGS:
            raise ValueError(
                f"unknown profile key {key!r}; expected one of {', '.join(_TUNABLE)}"
            )
    return values


def resolve_profile(profile=_DEFAULT_PROFILE, overrides=None):
    target = dict(_PROFILES[_DEFAULT_PROFILE])
    name = str(profile).strip() if profile is not None else _DEFAULT_PROFILE
    name = name or _DEFAULT_PROFILE
    if name in _PROFILES:
        target.update(_PROFILES[name])
    elif name.endswith(".json") or Path(name).is_file():
        target.update(_validate_keys(_load_json(name)))
    else:
        raise ValueError(_profile_error(name))
    target = _apply_overrides(target, overrides)
    return {key: _normalize(key, target[key]) for key in _TUNABLE}


class Tuner:
    def __init__(self, root="/"):
        self.root = Path(root)

    def _path(self, value):
        path = Path(str(value))
        if self.root != Path("/"):
            path = self.root / path.relative_to("/")
        return path

    def _glob(self, pattern):
        try:
            found = glob.glob(str(self._path(pattern)))
        except OSError:
            found = []
        return [Path(p) for p in sorted(found)]

    def _freq_pairs(self):
        pairs = []
        for minimum in self._glob(_FREQ_MIN):
            maximum = minimum.with_name("scaling_max_freq")
            if maximum.exists():
                pairs.append((minimum, maximum))
        return pairs

    def paths(self, name):
        if name == "cpu_frequency":
            return [path for pair in self._freq_pairs() for path in pair]
        return self._glob(_SETTINGS[name]["path"])

    def _read(self, path):
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError:
            return None

    def _read_int(self, path):
        value = self._read(path)
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            return None

    def _write(self, path, text):
        path.write_text(text, encoding="utf-8")

    def _current(self, name, path):
        raw = self._read(path)
        return _decode(name, raw) if raw is not None else None

    def _supported(self, name, value, paths):
        available = _SETTINGS[name].get("available")
        target = _display(name, value)
        if available:
            for path in paths:
                text = self._read(path.with_name(available))
                if text is not None and target not in text.split():
                    return False
        if name == "smt":
            for path in paths:
                if self._current(name, path) == "notsupported":
                    return False
        return True

    def _failure_status(self, failures):
        denied = all(
            isinstance(ex, PermissionError)
            or getattr(ex, "errno", None) in _PERMISSION_ERRNOS
            for ex in failures
        )
        return "denied" if denied else "failed"

    def _freq_targets(self, value, minimum, maximum):
        if isinstance(value, int):
            return value, value
        directory = minimum.parent
        low = self._read_int(directory / "cpuinfo_min_freq")
        high = self._read_int(directory / "cpuinfo_max_freq")
        if value == "range":
            return low, high
        if value == "min":
            return low, low
        if value == "max":
            return high, high
        base = self._read_int(directory / "base_frequency")
        if base is None:
            base = high
        return base, base

    def _freq_write(self, minimum, maximum, low, high):
        current_low = self._read_int(minimum)
        if current_low is not None and high < current_low:
            self._write(minimum, str(low))
            self._write(maximum, str(high))
        else:
            self._write(maximum, str(high))
            self._write(minimum, str(low))

    def _freq_current(self, minimum, maximum):
        low = self._read_int(minimum)
        high = self._read_int(maximum)
        return f"{_freq_text(low)}..{_freq_text(high)}"

    def _frequency_row(self, value, dry_run):
        name = "cpu_frequency"
        target = _display(name, value)
        pairs = self._freq_pairs()
        if not pairs:
            return {
                "setting": name,
                "path": str(_SETTINGS[name]["path"]),
                "current": "",
                "target": target,
                "status": "unavailable",
            }
        current = []
        resolved = []
        for minimum, maximum in pairs:
            text = self._freq_current(minimum, maximum)
            if text not in current:
                current.append(text)
            low, high = self._freq_targets(value, minimum, maximum)
            resolved.append((minimum, maximum, low, high))
        row = {
            "setting": name,
            "path": str(_SETTINGS[name]["path"]),
            "current": ";".join(current),
            "target": target,
            "status": "planned",
        }
        if any(low is None or high is None for _, _, low, high in resolved):
            row["status"] = "failed"
            return row
        if dry_run:
            return row
        if all(
            self._read_int(minimum) == low and self._read_int(maximum) == high
            for minimum, maximum, low, high in resolved
        ):
            row["status"] = "unchanged"
            return row
        failures = []
        for minimum, maximum, low, high in resolved:
            try:
                self._freq_write(minimum, maximum, low, high)
            except OSError as ex:
                failures.append(ex)
        if failures:
            row["status"] = self._failure_status(failures)
        else:
            row["status"] = (
                "tuned"
                if all(
                    self._read_int(minimum) == low and self._read_int(maximum) == high
                    for minimum, maximum, low, high in resolved
                )
                else "failed"
            )
        return row

    def _row(self, name, value, dry_run):
        if name == "cpu_frequency":
            return self._frequency_row(value, dry_run)
        paths = self.paths(name)
        target = _display(name, value)
        if not paths:
            return {
                "setting": name,
                "path": str(_SETTINGS[name]["path"]),
                "current": "",
                "target": target,
                "status": "unavailable",
            }
        current = []
        for path in paths:
            text = self._current(name, path) or ""
            if text not in current:
                current.append(text)
        if not self._supported(name, value, paths):
            return {
                "setting": name,
                "path": str(_SETTINGS[name]["path"]),
                "current": ",".join(current),
                "target": target,
                "status": "unsupported",
            }
        encoded = _encode(name, value)
        if dry_run:
            status = "planned"
        elif all(self._current(name, path) == target for path in paths):
            status = "unchanged"
        else:
            failures = []
            for path in paths:
                try:
                    self._write(path, encoded)
                except OSError as ex:
                    failures.append(ex)
            if failures:
                status = self._failure_status(failures)
            else:
                status = (
                    "tuned"
                    if all(self._current(name, path) == target for path in paths)
                    else "failed"
                )
        return {
            "setting": name,
            "path": str(_SETTINGS[name]["path"]),
            "current": ",".join(current),
            "target": target,
            "status": status,
        }

    def apply(self, profile=_DEFAULT_PROFILE, overrides=None, dry_run=False):
        resolved = resolve_profile(profile, overrides)
        rows = [self._row(name, resolved[name], dry_run) for name in _TUNABLE]
        return pd.DataFrame(rows, columns=list(_COLUMNS))


def tune(profile=_DEFAULT_PROFILE, overrides=None, root="/", dry_run=False):
    return Tuner(root).apply(profile=profile, overrides=overrides, dry_run=dry_run)
