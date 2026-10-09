"""The CPU backend built per CPU generation (x86-64 wheels; no models).

These tests call ggml directly to check the packaging: that every variant
ships, and that the ggml-cpu proxy picks the one ggml's own loader would.
The bindings themselves only use the NeMo-Speech.cpp C ABI.
"""

import ctypes
import os
import platform
import subprocess
import sys
import textwrap

import pytest

import nemo_speech
from nemo_speech import _loader

X86_64 = platform.machine().lower() in ("amd64", "x86_64") and sys.platform in ("win32", "linux")

# ggml's GGML_CPU_ALL_VARIANTS list for x86 at the pinned llama.cpp. MSVC
# builds no variant that needs F16C or FMA without AVX2, BF16 or AMX.
MSVC_VARIANTS = {
    "x64", "sse42", "sandybridge", "haswell", "skylakex", "cannonlake", "cascadelake", "icelake",
    "alderlake",
}
GCC_VARIANTS = MSVC_VARIANTS | {"ivybridge", "piledriver", "cooperlake", "zen4", "sapphirerapids"}

GGML_BACKEND_DEVICE_TYPE_CPU = 0

needs_variants = pytest.mark.skipif(
    not nemo_speech.build_info().get("cpu_variants"), reason="CPU backend not built per CPU generation"
)


def test_x86_64_wheels_build_cpu_variants():
    if os.environ.get("NEMO_SPEECH_LIB_PATH"):
        pytest.skip("libraries from NEMO_SPEECH_LIB_PATH")
    assert nemo_speech.build_info()["cpu_variants"] == X86_64


def _library(stem):
    directory = _loader.library_dir()
    names = [f"{stem}.dll"] if sys.platform == "win32" else [f"lib{stem}.so.0", f"lib{stem}.so"]
    for name in names:
        if (directory / name).exists():
            return directory / name
    pytest.fail(f"{stem} not found in {directory}")


def _variant_files():
    if sys.platform == "win32":
        prefix, suffix = "ggml-cpu-", ".dll"
    else:
        prefix, suffix = "libggml-cpu-", ".so"
    files = sorted(_loader.library_dir().glob(f"{prefix}*{suffix}"))
    return {f.name[len(prefix):-len(suffix)]: f for f in files}


def _module_of(address):
    """The file of the loaded library that contains ``address``."""
    if sys.platform == "win32":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = ctypes.c_void_p()
        # GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | ..._UNCHANGED_REFCOUNT
        assert kernel32.GetModuleHandleExW(0x4 | 0x2, ctypes.c_void_p(address), ctypes.byref(handle))
        buffer = ctypes.create_unicode_buffer(32768)
        assert kernel32.GetModuleFileNameW(handle, buffer, len(buffer))
        return buffer.value

    class DlInfo(ctypes.Structure):
        _fields_ = [("dli_fname", ctypes.c_char_p), ("dli_fbase", ctypes.c_void_p),
                    ("dli_sname", ctypes.c_char_p), ("dli_saddr", ctypes.c_void_p)]

    info = DlInfo()
    assert ctypes.CDLL("libdl.so.2").dladdr(ctypes.c_void_p(address), ctypes.byref(info))
    return os.fsdecode(info.dli_fname)


def _proxy():
    _loader.load_library("nemo_speech_asr_c")  # DLL directories, then ggml as the bindings load it
    proxy = ctypes.CDLL(str(_library("ggml-cpu")))
    proxy.ggml_backend_cpu_reg.restype = ctypes.c_void_p
    return proxy


def _expected_variant(variants):
    """The variant ggml's loader would pick: the first one with the highest score."""
    best, best_score = None, 0
    for name, path in variants.items():
        library = ctypes.CDLL(str(path))
        score = library.ggml_backend_score()
        if score > best_score:
            best, best_score = name, score
    return best


@needs_variants
def test_every_variant_ships():
    expected = MSVC_VARIANTS if sys.platform == "win32" else GCC_VARIANTS
    assert set(_variant_files()) == expected


@needs_variants
def test_variants_export_the_loader_entry_points():
    for name, path in _variant_files().items():
        library = ctypes.CDLL(str(path))
        assert hasattr(library, "ggml_backend_score"), name
        assert hasattr(library, "ggml_backend_init"), name
    # The baseline variant runs on every x86-64 CPU.
    assert ctypes.CDLL(str(_variant_files()["x64"])).ggml_backend_score() > 0


@needs_variants
def test_proxy_is_not_a_loadable_backend():
    proxy = _proxy()
    assert not hasattr(proxy, "ggml_backend_init")
    assert not hasattr(proxy, "ggml_backend_score")


@needs_variants
def test_proxy_loads_the_best_variant():
    proxy = _proxy()
    reg = proxy.ggml_backend_cpu_reg()
    assert reg
    variants = _variant_files()
    assert os.path.samefile(_module_of(reg), variants[_expected_variant(variants)])


@needs_variants
def test_registry_uses_the_proxy():
    proxy = _proxy()
    ggml = ctypes.CDLL(str(_library("ggml")))
    base = ctypes.CDLL(str(_library("ggml-base")))
    ggml.ggml_backend_dev_by_type.restype = ctypes.c_void_p
    ggml.ggml_backend_dev_by_type.argtypes = [ctypes.c_int]
    base.ggml_backend_dev_backend_reg.restype = ctypes.c_void_p
    base.ggml_backend_dev_backend_reg.argtypes = [ctypes.c_void_p]
    base.ggml_backend_dev_init.restype = ctypes.c_void_p
    base.ggml_backend_dev_init.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    base.ggml_backend_free.argtypes = [ctypes.c_void_p]
    proxy.ggml_backend_is_cpu.restype = ctypes.c_bool
    proxy.ggml_backend_is_cpu.argtypes = [ctypes.c_void_p]
    proxy.ggml_backend_cpu_set_n_threads.argtypes = [ctypes.c_void_p, ctypes.c_int]

    device = ggml.ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)
    assert device
    assert base.ggml_backend_dev_backend_reg(device) == proxy.ggml_backend_cpu_reg()
    backend = base.ggml_backend_dev_init(device, None)
    assert backend
    try:
        assert proxy.ggml_backend_is_cpu(backend)
        proxy.ggml_backend_cpu_set_n_threads(backend, 2)
    finally:
        base.ggml_backend_free(backend)


_SELECTED_VARIANT = textwrap.dedent(
    """
    import ctypes, sys
    sys.path[:0] = [sys.argv[1]]
    import test_cpu_variants as t
    print(t._module_of(t._proxy().ggml_backend_cpu_reg()))
    """
)


def _selected_variant(tmp_path, requested):
    env = dict(os.environ, NEMO_SPEECH_CPU_VARIANT=requested)
    # Outside the repository, so the subprocess imports the installed package.
    result = subprocess.run(
        [sys.executable, "-c", _SELECTED_VARIANT, os.path.dirname(__file__)],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=True,
    )
    return result.stdout.strip(), result.stderr


@needs_variants
def test_variant_can_be_requested(tmp_path):
    selected, _ = _selected_variant(tmp_path, "x64")
    assert os.path.samefile(selected, _variant_files()["x64"])


@needs_variants
def test_unknown_variant_falls_back(tmp_path):
    selected, stderr = _selected_variant(tmp_path, "no-such-cpu")
    variants = _variant_files()
    assert os.path.samefile(selected, variants[_expected_variant(variants)])
    assert "NEMO_SPEECH_CPU_VARIANT=no-such-cpu" in stderr
