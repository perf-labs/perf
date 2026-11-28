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

import argparse
import configparser
import os
from pathlib import Path

_PERFCONFIG = ".perfconfig"
_PERFCONFIG_COMMENTS = ("#", ";")
_OUTPUT_STDOUT = frozenset({"-", "true", "1", "yes", "on", "stdout"})
_OUTPUT_DISABLED = frozenset({"false", "0", "no", "off", "none"})
_DEFAULT_OUTPUT_TEMPLATE = "{time}-{command}-{target}-{id}.json"


def output_targets(value):
    if value is None or value is False:
        return []
    if value is True:
        return [True]
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(output_targets(item))
        return out
    text = str(value).strip()
    low = text.casefold()
    if not text or low in _OUTPUT_STDOUT:
        return [True]
    if low in _OUTPUT_DISABLED:
        return []
    return [text]


def has_output(value):
    return bool(output_targets(value))


def show_table(value, from_config=False):
    targets = output_targets(value)
    if not targets:
        return True
    return bool(from_config) and not any(target is True for target in targets)


def output_template(value):
    targets = output_targets(value)
    if len(targets) != 1 or targets[0] is True:
        return None
    text = str(targets[0]).strip()
    if not text or text.endswith(("/", os.sep)) or text in (".", ".."):
        return None
    path = Path(text)
    try:
        if path.is_dir():
            return None
    except OSError:
        pass
    if path.is_absolute() or path.parent != Path("."):
        return None
    return text


class _PerfconfigDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entries = []

    def __setitem__(self, key, value):
        if key not in self or isinstance(value, (list, type(None))):
            self.entries.append((key, value))
        super().__setitem__(key, value)


def _perfconfig_value(raw):
    if raw is None:
        return ""
    if isinstance(raw, list):
        return "\n".join(str(part) for part in raw).strip()
    return str(raw).strip()


def _perfconfig_path():
    try:
        current = Path.cwd().resolve()
        home = Path.home().resolve()
    except OSError:
        return None
    while True:
        candidate = current / _PERFCONFIG
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            pass
        if current == home or current.parent == current:
            return None
        current = current.parent


def perfconfig_entries(section, path=None):
    path = Path(path) if path is not None else _perfconfig_path()
    if path is None or not path.is_file():
        return {}
    parser = configparser.ConfigParser(
        interpolation=None,
        strict=False,
        comment_prefixes=_PERFCONFIG_COMMENTS,
        inline_comment_prefixes=_PERFCONFIG_COMMENTS,
        dict_type=_PerfconfigDict,
    )
    try:
        with open(path, encoding="utf-8") as fh:
            parser.read_file(fh, source=str(path))
    except (configparser.Error, UnicodeDecodeError, OSError):
        return {}
    entries = {}
    if not parser.has_section(section):
        return entries
    node = parser._sections.get(section)
    history = getattr(node, "entries", None)
    if history is None:
        history = [(key, value) for key, value in parser.items(section)]
    for key, raw in history:
        value = _perfconfig_value(raw)
        if not key.strip() or not value:
            continue
        if key in entries:
            previous = entries[key]
            if not isinstance(previous, list):
                entries[key] = [previous]
            entries[key].append(value)
        else:
            entries[key] = value
    return entries


def _known(parser, key, dotted_keys=()):
    if dotted_keys and key.startswith(dotted_keys):
        return True
    return any(
        action.dest == key or key in {o.lstrip("-") for o in action.option_strings}
        for action in parser._actions
    )


def _aliases(parser, key, dotted_keys=()):
    if dotted_keys and key.startswith(dotted_keys):
        return {key}
    aliases = {key}
    for action in parser._actions:
        if action.dest == key:
            aliases |= {o.lstrip("-") for o in action.option_strings}
    return aliases


def _option_name(parser, key):
    for action in parser._actions:
        if action.dest == key or key in {o.lstrip("-") for o in action.option_strings}:
            for option in action.option_strings:
                if option.startswith("--"):
                    return option.lstrip("-")
            if action.option_strings:
                return action.option_strings[0].lstrip("-")
    return key


def _flag_only(parser, key):
    return any(
        (
            action.dest == key
            or _option_name(parser, key)
            in {o.lstrip("-") for o in action.option_strings}
        )
        and isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction))
        for action in parser._actions
    )


def _given(argv):
    given = set()
    for arg in argv:
        if arg == "--":
            break
        if arg.startswith("--"):
            given.add(arg[2:].split("=", 1)[0])
        elif arg.startswith("-") and len(arg) > 1:
            given.add(arg[:2].lstrip("-"))
    return given


def _truthy(value):
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    return None


def apply_perfconfig(
    parser,
    argv,
    command,
    config_path=None,
    *,
    section=None,
    options=None,
    prepend=False,
    skip=None,
    dotted_keys=(),
):
    options = options or parser
    options.set_defaults(_output_from_config=False)
    entries = perfconfig_entries(section or command, config_path)
    if not entries:
        return list(argv)
    argv = list(argv)
    given = _given(argv)
    pending = []
    for key, values in entries.items():
        if not isinstance(values, list):
            values = [values]
        for value in values:
            if skip is not None and skip(key, value, command, given):
                continue
            if key == "event" and "list" in given and not ({"e", "event"} & given):
                continue
            if key == "output":
                template = output_template(value)
                if template is not None:
                    options.set_defaults(_output_template=template)
                if not _known(options, key, dotted_keys) or (
                    _aliases(options, key, dotted_keys) & given
                ):
                    continue
                options.set_defaults(_output_from_config=True)
                pending.append(f"--output={value}")
                continue
            if not _known(options, key, dotted_keys) or (
                _aliases(options, key, dotted_keys) & given
            ):
                continue
            if _flag_only(options, key):
                on = _truthy(value)
                if on is None:
                    parser.error(
                        f"invalid [{command}] {key} = {value!r}; expected a boolean "
                        "(true/false)"
                    )
                if on:
                    pending.append(f"--{_option_name(options, key)}")
                continue
            pending.append(f"--{_option_name(options, key)}={value}")
    if not pending:
        return list(argv)
    if prepend:
        return [argv[0], *pending, *argv[1:]] if argv else pending
    return [*pending, *argv]


def _has_dict(value):
    if isinstance(value, dict):
        return True
    if isinstance(value, (list, tuple)):
        return any(_has_dict(v) for v in value)
    return False


def _add_flat(out, column, value):
    if column not in out:
        out[column] = [value]
    elif isinstance(out[column], list):
        out[column].append(value)
    else:
        out[column] = [out[column], value]


def _dict_entries(values):
    out = []
    if isinstance(values, dict):
        return [values]
    for value in values if isinstance(values, (list, tuple)) else ():
        out.extend(_dict_entries(value))
    return out


def _flatten_config(prefix, node, out):
    for key, value in dict(node or {}).items():
        full = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            _flatten_config(full, value, out)
        elif _has_dict(value):
            for alt in _dict_entries(value):
                sub = {}
                _flatten_config("", alt, sub)
                for name, val in sub.items():
                    _add_flat(out, f"{full}.{name}", val)
        else:
            out[full] = value
    return out


def normalize_config(config):
    return _flatten_config("", config, {})


_normalize_config = normalize_config


def _deep_merge(base, override):
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override
    out = dict(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _config_backend(config):
    if isinstance(config, dict):
        return config.get("backend")
    return getattr(config, "backend", None)


def _config_flag(config, name):
    if isinstance(config, dict):
        return bool(config.get(name, False))
    return bool(getattr(config, name, False))


config_backend = _config_backend
config_flag = _config_flag
add_flat = _add_flat
aliases = _aliases
deep_merge = _deep_merge
dict_entries = _dict_entries
flatten_config = _flatten_config
given_options = _given
has_dict = _has_dict
is_flag_only = _flag_only
is_known = _known
option_name = _option_name
truthy = _truthy

__all__ = [
    "add_flat",
    "aliases",
    "apply_perfconfig",
    "config_backend",
    "config_flag",
    "deep_merge",
    "dict_entries",
    "flatten_config",
    "given_options",
    "has_dict",
    "has_output",
    "is_flag_only",
    "is_known",
    "normalize_config",
    "option_name",
    "output_targets",
    "output_template",
    "perfconfig_entries",
    "show_table",
    "truthy",
]
