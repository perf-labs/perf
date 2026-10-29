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
import functools
import glob
import json
import os
import re
import shutil
import subprocess
import threading
from pathlib import Path

import pandas as pd

_COLUMNS = ("setting", "path", "current", "target", "status")
_DEFAULT_PROFILE = "default"
_TUNE_LOCK = threading.RLock()


def _serialized(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _TUNE_LOCK:
            return fn(*args, **kwargs)

    return wrapper


_FREQ_MIN = "/sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq"
_MSR_ENABLE = 0x1A0
_MSR_TURBO_BIT = 38
_TUNABLE = (
    "aslr",
    "governor",
    "energy_performance_preference",
    "cpu_frequency",
    "turbo",
    "turbo_msr",
    "boost",
    "smt",
    "nmi_watchdog",
    "watchdog",
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
    "kptr_restrict",
    "sched_autogroup_enabled",
    "sched_rt_runtime_us",
    "irqbalance",
    "irq_affinity",
    "power_supply",
    "isolcpus",
    "rcu_nocbs",
    "nohz_full",
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
    "watchdog": {
        "path": "/proc/sys/kernel/watchdog",
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
        "max": 4,
    },
    "kptr_restrict": {
        "path": "/proc/sys/kernel/kptr_restrict",
        "kind": "int",
        "min": 0,
        "max": 2,
    },
    "sched_autogroup_enabled": {
        "path": "/proc/sys/kernel/sched_autogroup_enabled",
        "kind": "bool",
    },
    "sched_rt_runtime_us": {
        "path": "/proc/sys/kernel/sched_rt_runtime_us",
        "kind": "int",
        "min": -1,
        "max": 1000000,
    },
    "turbo_msr": {
        "path": "/dev/cpu/*/msr",
        "kind": "msr_turbo",
    },
    "irqbalance": {
        "path": "systemd:irqbalance",
        "kind": "service",
        "service": "irqbalance",
    },
    "irq_affinity": {
        "path": "/proc/irq/default_smp_affinity",
        "kind": "irq_affinity",
    },
    "power_supply": {
        "path": "/sys/class/power_supply",
        "kind": "power",
        "choices": ("ac", "battery", "any"),
    },
    "isolcpus": {
        "path": "/proc/cmdline",
        "kind": "kernel_arg",
        "arg": "isolcpus",
    },
    "rcu_nocbs": {
        "path": "/proc/cmdline",
        "kind": "kernel_arg",
        "arg": "rcu_nocbs",
    },
    "nohz_full": {
        "path": "/sys/devices/system/cpu/nohz_full",
        "kind": "kernel_arg",
        "arg": "nohz_full",
    },
}
_PROFILES = {
    "default": {
        "aslr": True,
        "governor": "powersave",
        "energy_performance_preference": "balance_performance",
        "cpu_frequency": "range",
        "turbo": True,
        "turbo_msr": True,
        "boost": True,
        "smt": True,
        "nmi_watchdog": True,
        "watchdog": True,
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
        "kptr_restrict": 1,
        "sched_autogroup_enabled": True,
        "sched_rt_runtime_us": 950000,
        "irqbalance": True,
        "irq_affinity": False,
        "power_supply": "ac",
        "isolcpus": False,
        "rcu_nocbs": False,
        "nohz_full": False,
    },
    "latency": {
        "aslr": False,
        "governor": "performance",
        "energy_performance_preference": "performance",
        "cpu_frequency": "max",
        "turbo": False,
        "turbo_msr": False,
        "boost": False,
        "smt": False,
        "nmi_watchdog": False,
        "watchdog": False,
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
        "kptr_restrict": 0,
        "sched_autogroup_enabled": False,
        "sched_rt_runtime_us": -1,
        "irqbalance": False,
        "irq_affinity": True,
        "power_supply": "ac",
        "isolcpus": True,
        "rcu_nocbs": True,
        "nohz_full": False,
    },
    "throughput": {
        "aslr": False,
        "governor": "performance",
        "energy_performance_preference": "performance",
        "cpu_frequency": "max",
        "turbo": True,
        "turbo_msr": True,
        "boost": True,
        "smt": True,
        "nmi_watchdog": False,
        "watchdog": False,
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
        "kptr_restrict": 0,
        "sched_autogroup_enabled": True,
        "sched_rt_runtime_us": 950000,
        "irqbalance": False,
        "irq_affinity": False,
        "power_supply": "ac",
        "isolcpus": False,
        "rcu_nocbs": False,
        "nohz_full": False,
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
_THP_KEYS = ("thp", "thp_defrag")
_FREQ_KEYS = _THP_KEYS


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

    def paths(self, name):
        if name == "cpu_frequency":
            return [path for pair in self._freq_pairs() for path in pair]
        return self._glob(_SETTINGS[name]["path"])

    @_serialized
    def apply(self, profile=_DEFAULT_PROFILE, overrides=None, dry_run=False):
        resolved = resolve_profile(profile, overrides)
        rows = [self._row(name, resolved[name], dry_run) for name in _TUNABLE]
        frame = pd.DataFrame(rows, columns=list(_COLUMNS))
        warnings = self._compat_warnings(resolved)
        if warnings:
            frame.attrs["warnings"] = list(warnings)
        return frame

    def _compat_warnings(self, resolved):
        warnings = []
        try:
            nohz_on = bool(resolved.get("nohz_full", False))
        except Exception:
            nohz_on = False
        if nohz_on:
            intel_pstate = self._path("/sys/devices/system/cpu/intel_pstate/no_turbo")
            try:
                has_pstate = intel_pstate.exists()
            except OSError:
                has_pstate = False
            if has_pstate:
                warnings.append(
                    "nohz_full is incompatible with the intel_pstate driver "
                    "(see https://bugzilla.redhat.com/show_bug.cgi?id=1378529); "
                    "fix the CPU frequency or disable nohz_full"
                )
        return warnings

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
                raw = self._read(path)
                if raw is not None and raw.strip() == "notsupported":
                    return False
                if (
                    raw is not None
                    and raw.strip() == "forceoff"
                    and bool(value) is True
                ):
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
        already = all(
            self._read_int(minimum) == low and self._read_int(maximum) == high
            for minimum, maximum, low, high in resolved
        )
        if already:
            row["status"] = "unchanged"
            return row
        if dry_run:
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

    def _service_state(self, service):
        systemctl = shutil.which("systemctl")
        if systemctl is not None:
            try:
                load = subprocess.run(
                    [systemctl, "show", "--property=LoadState", "--value", service],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except (OSError, subprocess.SubprocessError):
                pass
            else:
                if load.returncode != 0:
                    pass
                elif load.stdout.strip() != "loaded":
                    pass
                else:
                    try:
                        active = subprocess.run(
                            [
                                systemctl,
                                "show",
                                "--property=ActiveState",
                                "--value",
                                service,
                            ],
                            capture_output=True,
                            text=True,
                            timeout=10,
                        )
                    except (OSError, subprocess.SubprocessError):
                        pass
                    else:
                        if active.returncode == 0:
                            value = active.stdout.strip()
                            if value in ("active", "activating", "reloading"):
                                return True
                            if value in (
                                "inactive",
                                "deactivating",
                                "failed",
                                "dead",
                            ):
                                return False
                            return None
        service_cmd = shutil.which("service")
        if service_cmd is not None:
            try:
                proc = subprocess.run(
                    [service_cmd, service, "status"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except (OSError, subprocess.SubprocessError):
                return None
            combined = f"{proc.stdout or ''}\n{proc.stderr or ''}".lower()
            if "unrecognized" in combined or "not found" in combined:
                return None
            if proc.returncode == 0:
                return True
            return False
        return None

    def _service_set(self, service, enabled):
        action = "start" if enabled else "stop"
        errors = []

        def _failure(detail, returncode=None):
            low = str(detail).lower()
            if "permission" in low or "access denied" in low or "not authorized" in low:
                return PermissionError(errno.EACCES, detail)
            if returncode is not None and returncode != 1:
                return OSError(returncode, detail)
            return OSError(detail)

        systemctl = shutil.which("systemctl")
        if systemctl is not None:
            try:
                proc = subprocess.run(
                    [systemctl, action, service],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except (OSError, subprocess.SubprocessError) as ex:
                errors.append(ex if isinstance(ex, OSError) else OSError(str(ex)))
            else:
                if proc.returncode == 0:
                    return
                detail = proc.stderr.strip() or proc.stdout.strip() or action
                errors.append(_failure(detail, proc.returncode))
        service_cmd = shutil.which("service")
        if service_cmd is not None:
            try:
                proc = subprocess.run(
                    [service_cmd, service, action],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except (OSError, subprocess.SubprocessError) as ex:
                errors.append(ex if isinstance(ex, OSError) else OSError(str(ex)))
            else:
                if proc.returncode == 0:
                    return
                detail = proc.stderr.strip() or proc.stdout.strip() or action
                errors.append(_failure(detail, proc.returncode))
        if errors:
            raise errors[0]
        raise OSError(errno.ENOENT, "neither systemctl nor service found")

    def _readonly_row(self, name, value, current, available=True):
        target = _display(name, value)
        if not available:
            return {
                "setting": name,
                "path": str(_SETTINGS[name]["path"]),
                "current": "",
                "target": target,
                "status": "unavailable",
            }
        status = "unchanged" if current == target else "unsupported"
        return {
            "setting": name,
            "path": str(_SETTINGS[name]["path"]),
            "current": current,
            "target": target,
            "status": status,
        }

    def _service_row(self, name, value, dry_run):
        service = _SETTINGS[name]["service"]
        if self.root != Path("/"):
            return self._readonly_row(name, value, "", False)
        current = self._service_state(service)
        row = self._readonly_row(
            name, value, _display(name, current), current is not None
        )
        if current is None:
            return row
        if current == bool(value):
            row["status"] = "unchanged"
            return row
        if dry_run:
            row["status"] = "planned"
            return row
        try:
            self._service_set(service, bool(value))
        except OSError as ex:
            row["status"] = self._failure_status([ex])
            return row
        row["status"] = (
            "tuned" if self._service_state(service) == bool(value) else "failed"
        )
        return row

    def _power_current(self):
        path = self._path(_SETTINGS["power_supply"]["path"])
        try:
            entries = list(path.iterdir())
        except OSError:
            return None
        for entry in entries:
            if self._read(entry / "type") != "Mains":
                continue
            online = self._read(entry / "online")
            if online == "1":
                return "ac"
            if online == "0":
                return "battery"
        return None

    def _power_row(self, name, value):
        current = self._power_current()
        if current is None:
            return self._readonly_row(name, value, current, False)
        if str(value) == "any":
            row = self._readonly_row(name, value, current, True)
            row["status"] = "unchanged"
            return row
        return self._readonly_row(name, value, current, True)

    def _kernel_arg_current(self, name):
        spec = _SETTINGS[name]
        path = self._path(spec["path"])
        text = self._read(path)
        if text is None:
            return None
        if name == "nohz_full":
            stripped = text.strip()
            if stripped in ("", "0", "(null)"):
                return "off"
            return "on"
        match = re.search(rf"\b{re.escape(spec['arg'])}=([^\s]+)", text)
        return "on" if match and match.group(1) not in ("", "0") else "off"

    def _kernel_arg_row(self, name, value):
        current = self._kernel_arg_current(name)
        return self._readonly_row(name, value, current, current is not None)

    def _isolated_cpus(self):
        found = set()
        for text in (
            self._read(self._path("/sys/devices/system/cpu/isolated")) or "",
            self._read(self._path("/proc/cmdline")) or "",
        ):
            for match in re.finditer(
                r"\b(?:isolcpus|nohz_full)\s*=\s*([0-9,\-\s]+)", text
            ):
                for part in match.group(1).split(","):
                    part = part.strip()
                    if not part or part == "0":
                        continue
                    low, _, high = part.partition("-")
                    try:
                        start = int(low.strip())
                        end = int(high.strip()) if high.strip() else start
                    except ValueError:
                        continue
                    found.update(range(start, end + 1))
            bare = text.strip()
            if re.fullmatch(r"[0-9,\-\s]+", bare or "") and "isolcpus" not in text:
                for part in bare.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    low, _, high = part.partition("-")
                    try:
                        start = int(low.strip())
                        end = int(high.strip()) if high.strip() else start
                    except ValueError:
                        continue
                    found.update(range(start, end + 1))
        return found

    def _parse_affinity_mask(self, text):
        groups = [g.strip() for g in str(text).strip().split(",") if g.strip()]
        if not groups:
            return None
        mask = 0
        for group in groups:
            try:
                chunk = int(group, 16)
            except ValueError:
                return None
            mask = (mask << 32) | (chunk & 0xFFFFFFFF)
        return mask

    @staticmethod
    def _format_affinity_mask(mask):
        digits = f"{max(int(mask), 0):x}"
        groups = []
        while digits:
            groups.append(digits[-8:])
            digits = digits[:-8]
        return ",".join(reversed(groups)) or "0"

    def _irq_affinity_current(self):
        paths = self._irq_affinity_files()
        isolated = self._isolated_cpus()
        masks = []
        for path in paths:
            text = self._read(path)
            if text is None:
                continue
            mask = self._parse_affinity_mask(text)
            if mask is not None:
                masks.append(mask)
        if not masks:
            return None
        if not isolated:
            return "on"
        if any(mask & (1 << cpu) for mask in masks for cpu in isolated):
            return "off"
        return "on"

    def _irq_affinity_files(self):
        seen = {}
        default = self._path(_SETTINGS["irq_affinity"]["path"])
        if default.exists():
            seen[str(default)] = default
        for path in self._glob("/proc/irq/[0-9]*/smp_affinity"):
            seen[str(path)] = path
        return sorted(seen.values(), key=str)

    def _irq_affinity_row(self, name, value, dry_run=False):
        path = self._path(_SETTINGS[name]["path"])
        if not path.exists():
            return self._readonly_row(name, value, "", False)
        current = self._irq_affinity_current()
        if current is None:
            return self._readonly_row(name, value, current, False)
        if not value:
            row = self._readonly_row(name, value, current, True)
            row["status"] = "unchanged"
            return row
        isolated = self._isolated_cpus()
        if not isolated:
            return self._readonly_row(name, value, current, True)
        if current == "on":
            return self._readonly_row(name, value, current, True)
        if dry_run:
            row = self._readonly_row(name, value, current, True)
            row["status"] = "planned"
            return row
        failures = []
        for target in self._irq_affinity_files():
            text = self._read(target)
            if text is None:
                continue
            mask = self._parse_affinity_mask(text)
            if mask is None:
                continue
            if all(not (mask & (1 << cpu)) for cpu in isolated):
                continue
            desired = mask
            for cpu in isolated:
                desired &= ~(1 << cpu)
            if desired == 0:
                continue
            try:
                self._write(target, self._format_affinity_mask(desired))
            except OSError as ex:
                failures.append(ex)
        row = self._readonly_row(name, value, self._irq_affinity_current(), True)
        if row["current"] is None or row["current"] == "":
            return self._readonly_row(name, value, current, False)
        if failures:
            row["status"] = self._failure_status(failures)
            return row
        if row["current"] == "on":
            row["status"] = "tuned"
        else:
            row["status"] = "failed"
        return row

    def _msr_value(self, path):
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            raise
        try:
            try:
                data = os.pread(fd, 8, _MSR_ENABLE)
            except (AttributeError, OSError):
                try:
                    os.lseek(fd, _MSR_ENABLE, os.SEEK_SET)
                    data = os.read(fd, 8)
                except OSError:
                    raise
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
        if len(data) != 8:
            return None
        value = int.from_bytes(data, "little")
        return not bool(value & (1 << _MSR_TURBO_BIT))

    def _msr_write(self, path, enabled):
        fd = os.open(path, os.O_RDWR)
        try:
            try:
                data = os.pread(fd, 8, _MSR_ENABLE)
            except (AttributeError, OSError):
                os.lseek(fd, _MSR_ENABLE, os.SEEK_SET)
                data = os.read(fd, 8)
            if len(data) != 8:
                raise OSError(errno.EIO, "short MSR read")
            value = int.from_bytes(data, "little")
            mask = 1 << _MSR_TURBO_BIT
            value = value & ~mask if enabled else value | mask
            payload = value.to_bytes(8, "little")
            try:
                os.pwrite(fd, payload, _MSR_ENABLE)
            except (AttributeError, OSError):
                os.lseek(fd, _MSR_ENABLE, os.SEEK_SET)
                os.write(fd, payload)
        finally:
            os.close(fd)

    def _msr_turbo_row(self, name, value, dry_run):
        cpuinfo = self._read(self._path("/proc/cpuinfo")) or ""
        if "GenuineIntel" not in cpuinfo:
            return {
                "setting": name,
                "path": str(_SETTINGS[name]["path"]),
                "current": "",
                "target": _display(name, value),
                "status": "unavailable",
            }
        intel_pstate = self._path("/sys/devices/system/cpu/intel_pstate/no_turbo")
        if intel_pstate.exists():
            return {
                "setting": name,
                "path": str(_SETTINGS[name]["path"]),
                "current": "",
                "target": _display(name, value),
                "status": "unavailable",
            }
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
        failures = []
        unreadable = 0
        for path in paths:
            try:
                state = self._msr_value(path)
            except OSError as ex:
                failures.append(ex)
                continue
            if state is None:
                unreadable += 1
                continue
            label = _display(name, state)
            if label not in current:
                current.append(label)
        if failures or unreadable:
            if unreadable and not failures:
                failures.append(OSError(errno.EIO, "short MSR read"))
            return {
                "setting": name,
                "path": str(_SETTINGS[name]["path"]),
                "current": ",".join(current),
                "target": target,
                "status": self._failure_status(failures),
            }
        row = {
            "setting": name,
            "path": str(_SETTINGS[name]["path"]),
            "current": ",".join(current),
            "target": target,
            "status": "planned",
        }
        if len(current) == 1 and current[0] == target:
            row["status"] = "unchanged"
            return row
        if dry_run:
            return row
        for path in paths:
            try:
                if self._msr_value(path) != bool(value):
                    self._msr_write(path, bool(value))
            except OSError as ex:
                failures.append(ex)
        if failures:
            row["status"] = self._failure_status(failures)
        else:
            row["status"] = (
                "tuned"
                if all(self._msr_value(path) == bool(value) for path in paths)
                else "failed"
            )
        return row

    def _row(self, name, value, dry_run):
        if name == "cpu_frequency":
            return self._frequency_row(value, dry_run)
        if name == "turbo_msr":
            return self._msr_turbo_row(name, value, dry_run)
        if name == "irqbalance":
            return self._service_row(name, value, dry_run)
        if name == "irq_affinity":
            return self._irq_affinity_row(name, value, dry_run)
        if name == "power_supply":
            return self._power_row(name, value)
        if _SETTINGS[name]["kind"] == "kernel_arg":
            return self._kernel_arg_row(name, value)
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
        if all(self._current(name, path) == target for path in paths):
            status = "unchanged"
        elif dry_run:
            status = "planned"
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


@_serialized
def tune(profile=_DEFAULT_PROFILE, overrides=None, root="/", dry_run=False):
    return Tuner(root).apply(profile=profile, overrides=overrides, dry_run=dry_run)


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
    if kind in (
        "bool",
        "inverted_bool",
        "smt",
        "service",
        "msr_turbo",
        "kernel_arg",
        "irq_affinity",
    ):
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
        parsed = _parse_freq_khz(value)
        if parsed is not None and parsed > 0:
            return parsed
        number = _as_int(value)
        if number is not None and number > 0:
            return number
        raise ValueError(
            f"profile {name} needs min, max, base, range, or a positive "
            f"frequency in kHz (e.g. 2400000, '2.4GHz', '800MHz'), "
            f"got {value!r}"
        )
    if kind == "power":
        text = str(value).strip().lower()
        choices = spec.get("choices", ("ac", "battery", "any"))
        if text not in choices:
            raise ValueError(
                f"profile {name} must be one of "
                f"{', '.join(repr(c) for c in choices)}, got {value!r}"
            )
        return text
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
    if kind == "msr_turbo":
        return "0" if value else "1"
    if kind == "service":
        return "active" if value else "inactive"
    if kind == "kernel_arg":
        return "1" if value else "0"
    if kind == "irq_affinity":
        return "1" if value else "0"
    return str(value)


def _decode(name, raw):
    text = str(raw).strip()
    kind = _SETTINGS[name]["kind"]
    if kind == "aslr":
        return {"0": "off", "1": "partial", "2": "on"}.get(text, text)
    if kind == "bool":
        low = text.strip().lower()
        if low in ("1", "y", "yes", "true", "on", "enabled", "enable"):
            return "on"
        if low in ("0", "n", "no", "false", "off", "disabled", "disable"):
            return "off"
        return text
    if kind in ("inverted_bool", "msr_turbo"):
        low = text.strip().lower()
        if low in ("0", "n", "no", "false", "off", "disabled", "disable"):
            return "on"
        if low in ("1", "y", "yes", "true", "on", "enabled", "enable"):
            return "off"
        if text == "0":
            return "on"
        if text == "1":
            return "off"
        return text
    if kind == "smt":
        if text.strip() == "forceoff":
            return "off"
        found = _THP_CURRENT.search(text)
        return found.group(1) if found else text
    if kind in ("service", "kernel_arg", "irq_affinity"):
        low = text.strip().lower()
        if low in ("1", "active", "activating", "reloading", "enabled", "enable"):
            return "on"
        if low in (
            "0",
            "inactive",
            "deactivating",
            "failed",
            "dead",
            "disabled",
            "disable",
            "off",
            "false",
            "no",
        ):
            return "off"
        return text
    if name in _THP_KEYS:
        found = _THP_CURRENT.search(text)
        return found.group(1) if found else text
    return text


def _parse_freq_khz(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            number = int(value)
        except (TypeError, ValueError):
            return None
        return number if number > 0 else None
    text = str(value).strip().lower().replace(" ", "")
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(ghz|mhz|khz|hz)", text)
    if not match:
        return None
    number, unit = float(match.group(1)), match.group(2)
    factor = {"ghz": 1000000.0, "mhz": 1000.0, "khz": 1.0, "hz": 0.001}[unit]
    return int(number * factor)


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
