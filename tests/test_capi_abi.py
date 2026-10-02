"""Keep the hand-written ctypes mirror in sync with the vendored C headers.

The headers are append-only, so the likely drift is a new struct field or a
new function that the bindings have not picked up yet.
"""

import ctypes
import importlib
import re

import pytest

_STRUCT_RE = re.compile(r"typedef\s+struct\s+(\w+)\s*\{(.*?)\}\s*\1\s*;", re.S)
_FUNC_RE = re.compile(r"NEMO_SPEECH_\w+_API\b[^;{]*?\b(nemo_speech_\w+)\s*\(([^)]*)\)", re.S)

HEADERS = {
    "asr.h": "nemo_speech.capi.asr",
    "diar.h": "nemo_speech.capi.diar",
    "nmt.h": "nemo_speech.capi.nmt",
    "tts.h": "nemo_speech.capi.tts",
}


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def _header(upstream_dir, name: str) -> str:
    return _strip_comments((upstream_dir / "include" / "nemo_speech" / name).read_text("utf-8"))


def _header_structs(text: str) -> dict:
    structs = {}
    for name, body in _STRUCT_RE.findall(text):
        structs[name] = [re.findall(r"\w+", d)[-1] for d in body.split(";") if d.strip()]
    return structs


def _header_functions(text: str) -> dict:
    functions = {}
    for name, params in _FUNC_RE.findall(text):
        params = params.strip()
        functions[name] = 0 if params in ("", "void") else params.count(",") + 1
    return functions


@pytest.mark.parametrize("header, module_name", HEADERS.items())
def test_structs_match_header(upstream_dir, header, module_name):
    module = importlib.import_module(module_name)
    structs = _header_structs(_header(upstream_dir, header))
    assert structs, f"no structs parsed from {header}"
    for name, fields in structs.items():
        binding = getattr(module, name, None)
        assert binding is not None, f"struct {name} is not bound"
        assert [f[0] for f in binding._fields_] == fields, f"field mismatch in {name}"


@pytest.mark.parametrize("header, module_name", HEADERS.items())
def test_functions_match_header(upstream_dir, header, module_name):
    module = importlib.import_module(module_name)
    functions = _header_functions(_header(upstream_dir, header))
    assert functions, f"no functions parsed from {header}"
    for name, arity in functions.items():
        binding = getattr(module, name, None)
        assert binding is not None, f"function {name} is not bound"
        assert len(binding.argtypes) == arity, f"argument count mismatch in {name}"


@pytest.mark.parametrize(
    "module_name, default_fn, struct_name",
    [
        ("nemo_speech.capi.asr", "nemo_speech_asr_recognition_options_default",
         "nemo_speech_asr_recognition_options"),
        ("nemo_speech.capi.tts", "nemo_speech_tts_runtime_config_default",
         "nemo_speech_tts_runtime_config"),
        ("nemo_speech.capi.tts", "nemo_speech_tts_synthesis_options_default",
         "nemo_speech_tts_synthesis_options"),
        ("nemo_speech.capi.tts", "nemo_speech_tts_synthesis_stats_default",
         "nemo_speech_tts_synthesis_stats"),
    ],
)
def test_struct_sizes_match_library(module_name, default_fn, struct_name):
    """The library reports sizeof() of its own definition through *_default()."""
    module = importlib.import_module(module_name)
    native = getattr(module, default_fn)()
    assert native.size == ctypes.sizeof(getattr(module, struct_name))


@pytest.mark.parametrize(
    "module_name, version_fn",
    [
        ("nemo_speech.capi.asr", "nemo_speech_asr_version"),
        ("nemo_speech.capi.nmt", "nemo_speech_nmt_version"),
        ("nemo_speech.capi.tts", "nemo_speech_tts_version"),
    ],
)
def test_library_version_matches_vendored_source(upstream_dir, module_name, version_fn):
    match = re.search(
        r"^NEMO_SPEECH_VERSION:\s*(\S+)", (upstream_dir / "VERSION").read_text("utf-8"), re.M
    )
    assert match, "could not parse VERSION"
    version = getattr(importlib.import_module(module_name), version_fn)().decode()
    # Formatted as "nemo-speech-<component> <version>".
    assert version.split()[-1] == match.group(1)


def test_sized_structs_default_their_size():
    from nemo_speech.capi import asr, diar, nmt, tts

    for struct in (
        asr.nemo_speech_asr_recognizer_config,
        diar.nemo_speech_diar_model_config,
        nmt.nemo_speech_nmt_translator_config,
        tts.nemo_speech_tts_synthesizer_config,
    ):
        assert struct().size == ctypes.sizeof(struct)
