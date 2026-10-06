"""Keep the hand-written ctypes mirror in sync with the vendored C headers.

The headers are append-only, so the likely drift is a new struct field or a
new function that the bindings have not picked up yet. Beyond names, the
types of every field, argument and return value, the enum values and the
callback signature are compared: ctypes cannot detect a mismatch, and a wrong
width or pointer level silently corrupts memory.
"""

import ctypes
import importlib
import re

import pytest

_STRUCT_RE = re.compile(r"typedef\s+struct\s+(\w+)\s*\{(.*?)\}\s*\1\s*;", re.S)
_FUNC_RE = re.compile(r"NEMO_SPEECH_\w+_API\b[^;{]*?\b(nemo_speech_\w+)\s*\(([^)]*)\)", re.S)
_TYPED_FUNC_RE = re.compile(
    r"NEMO_SPEECH_\w+_API\s+([^;{(]*?)\s*\b(nemo_speech_\w+)\s*\(([^)]*)\)", re.S
)
_OPAQUE_RE = re.compile(r"typedef\s+struct\s+(\w+)\s+\1\s*;")
_ENUM_RE = re.compile(r"typedef\s+enum\s*\w*\s*\{(.*?)\}\s*(\w+)\s*;", re.S)
_CALLBACK_RE = re.compile(r"typedef\s+([^;(]*?)\(\s*\*\s*(\w+)\s*\)\s*\(([^)]*)\)\s*;", re.S)
_SCALARS = {
    "bool": ctypes.c_bool,
    "float": ctypes.c_float,
    "double": ctypes.c_double,
    "int": ctypes.c_int,
    "int32_t": ctypes.c_int32,
    "int64_t": ctypes.c_int64,
    "uint64_t": ctypes.c_uint64,
    "uint8_t": ctypes.c_uint8,
    "size_t": ctypes.c_size_t,
}

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


# ---- Types ----


def _split_declaration(declaration: str):
    """``"const char* const* phrases"`` -> ``("const char* const*", "phrases")``."""
    declaration = " ".join(declaration.split())
    name = re.findall(r"\w+", declaration)[-1]
    return declaration[: declaration.rindex(name)].strip(), name


def _expected_ctype(c_type: str, modules, opaque):
    """The ctypes type the bindings should use for a C type."""
    stars = c_type.count("*")
    words = re.sub(r"\bconst\b|\*", " ", c_type).split()
    assert len(words) == 1, c_type
    base = words[0]
    if base == "void" and stars == 0:
        return None
    if base == "char" and stars:
        expected, stars = ctypes.c_char_p, stars - 1
    elif (base == "void" or base in opaque) and stars:
        expected, stars = ctypes.c_void_p, stars - 1  # opaque handles are void*
    elif base in _SCALARS:
        expected = _SCALARS[base]
    else:  # enums (bound as c_int), structs and callback typedefs
        expected = next(getattr(m, base) for m in modules if hasattr(m, base))
    for _ in range(stars):
        expected = ctypes.POINTER(expected)
    return expected


def _modules():
    return [importlib.import_module(m) for m in HEADERS.values()]


def _opaque_types(upstream_dir):
    return {name for header in HEADERS for name in _OPAQUE_RE.findall(_header(upstream_dir, header))}


@pytest.mark.parametrize("header, module_name", HEADERS.items())
def test_struct_field_types_match_header(upstream_dir, header, module_name):
    module = importlib.import_module(module_name)
    modules, opaque = [module, *_modules()], _opaque_types(upstream_dir)
    for name, body in _STRUCT_RE.findall(_header(upstream_dir, header)):
        bound = dict(getattr(module, name)._fields_)
        for declaration in filter(str.strip, body.split(";")):
            c_type, field = _split_declaration(declaration)
            expected = _expected_ctype(c_type, modules, opaque)
            assert bound[field] is expected, f"{name}.{field}: {c_type} bound as {bound[field]}"


@pytest.mark.parametrize("header, module_name", HEADERS.items())
def test_function_types_match_header(upstream_dir, header, module_name):
    module = importlib.import_module(module_name)
    modules, opaque = [module, *_modules()], _opaque_types(upstream_dir)
    functions = _TYPED_FUNC_RE.findall(_header(upstream_dir, header))
    assert functions, f"no functions parsed from {header}"
    for return_type, name, params in functions:
        function = getattr(module, name)
        expected = _expected_ctype(" ".join(return_type.split()), modules, opaque)
        assert function.restype is expected, f"{name} returns {return_type}"
        params = params.strip()
        if params in ("", "void"):
            continue
        for i, param in enumerate(params.split(",")):
            c_type, arg = _split_declaration(param)
            expected = _expected_ctype(c_type, modules, opaque)
            assert function.argtypes[i] is expected, f"{name}({arg}): {c_type} bound as {function.argtypes[i]}"


@pytest.mark.parametrize("header, module_name", HEADERS.items())
def test_enum_values_match_header(upstream_dir, header, module_name):
    module = importlib.import_module(module_name)
    for body, enum_name in _ENUM_RE.findall(_header(upstream_dir, header)):
        assert getattr(module, enum_name) is ctypes.c_int, enum_name
        value = -1
        for item in filter(str.strip, body.split(",")):
            constant, _, explicit = (part.strip() for part in item.partition("="))
            value = int(explicit, 0) if explicit else value + 1
            assert getattr(module, constant) == value, f"{enum_name}: {constant}"


def test_callback_types_match_header(upstream_dir):
    modules, opaque = _modules(), _opaque_types(upstream_dir)
    callbacks = [cb for header in HEADERS for cb in _CALLBACK_RE.findall(_header(upstream_dir, header))]
    assert callbacks, "no callback typedefs parsed"
    for return_type, name, params in callbacks:
        bound = next(getattr(m, name) for m in modules if hasattr(m, name))
        assert bound._restype_ is _expected_ctype(return_type.strip(), modules, opaque), name
        expected = [_expected_ctype(_split_declaration(p)[0], modules, opaque) for p in params.split(",")]
        assert list(bound._argtypes_) == expected, name
