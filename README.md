# nemo-speech (Python)

> **Unofficial.** This project is not affiliated with, sponsored by, or
> endorsed by NVIDIA. It wraps the public NeMo-Speech.cpp C ABI as-is.

Python bindings for [NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp),
in the style of [llama-cpp-python](https://github.com/abetlen/llama-cpp-python):

- the C++ runtime is built from a vendored submodule (`vendor/NeMo-Speech.cpp`),
- its shared libraries ship inside the wheel (`nemo_speech/lib`),
- Python talks to the stable C ABI (`include/nemo_speech/*.h`) through `ctypes`,
  so one `py3-none-<platform>` wheel serves every Python 3 version.

All four C ABI families are bound: ASR (offline and streaming), standalone
speaker diarization, TTS (MagpieTTS) and NMT (Riva-Translate).

| Platform | Wheel tag | GPU | Status |
|---|---|---|---|
| Windows x64 | `win_amd64` | Vulkan | cibuildwheel from the sdist, delvewheel, all tests (CPU; Vulkan on Intel Iris Xe) |
| Linux x86_64 | `manylinux_2_28_x86_64` | — | cibuildwheel from the sdist in Docker, auditwheel; all E2E tests on Debian 12 |
| Linux aarch64 | `manylinux_2_28_aarch64` | — | CI: cibuildwheel from the sdist, auditwheel, tests without models |
| macOS arm64 | `macosx_13_0_arm64` | Metal | CI: cibuildwheel from the sdist, delocate, tests without models |
| Windows x64, CUDA 12.8 | `win_amd64` | CUDA | CI: all release architectures built and repaired (704 MB, about 90 minutes); not run on a GPU |
| Linux x86_64, CUDA 12.8 | `manylinux_2_28_x86_64` | CUDA | CI: all release architectures built, repaired and tested against the driver stub (732 MB, about 80 minutes); not run on a GPU |

CUDA wheels (Linux x86_64, Windows x64) build without a GPU and bundle
cuBLAS; see [CUDA](#cuda). They have not been run on an NVIDIA GPU yet.

## Layout

| Path | Purpose |
|---|---|
| `CMakeLists.txt` | Superbuild: SentencePiece → NeMo-Speech.cpp → install into `nemo_speech/lib` |
| `cmake/materialize_llama_cpp.cmake` | Applies upstream `patches/` to llama.cpp (see Known issues) |
| `cmake/install_shared_libs.cmake` | Installs ELF/Mach-O libraries once under their SONAME (wheels have no symlinks) |
| `cmake/project_include.cmake` | Compile settings injected into the external projects |
| `patches/llama.cpp/` | This repository's llama.cpp patches, applied after upstream's series |
| `nemo_speech/capi/` | Low-level ctypes mirror of the four C headers (C names unchanged) |
| `nemo_speech/{asr,diar,tts,nmt}.py` | High-level API |
| `nemo_speech/models.py` | Model catalog and downloads, sharing the CLI's cache |
| `vendor/` | `NeMo-Speech.cpp` and `sentencepiece` submodules |
| `tests/` | ABI drift checks against the vendored headers, unit and E2E tests |
| `.github/workflows/wheels.yml` | sdist → cibuildwheel for every platform, plus a Windows Vulkan wheel |

## Build from source

Requirements: Python 3.9+, CMake 3.26+, git, and a C++17 compiler
(Windows: Visual Studio 2022 or its Build Tools with the C++ workload).

```bash
git clone https://github.com/PNL-toshiyaishihara/nemo-speech-python.git && cd nemo-speech-python
git submodule update --init vendor/NeMo-Speech.cpp vendor/sentencepiece
git -C vendor/NeMo-Speech.cpp submodule update --init --depth 1 llama.cpp
pip install .
```

Only the `llama.cpp` submodule of NeMo-Speech.cpp is needed; the others
(gRPC protos, KenLM, Open JTalk, ...) belong to features the wheel does not
build. An sdist (`python -m build --sdist`) contains everything and builds
without git metadata.

The build tree is kept in `build/<wheel tag>/`, so rebuilds are incremental.
Upstream CMake options pass straight through:

On Windows, keep the build tree path short (the nested Vulkan shader-generator
project otherwise exceeds MAX_PATH): build from a short checkout path or pass
`-C build-dir=C:/b`.

```bash
CMAKE_ARGS="-DGGML_VULKAN=ON" pip install .     # needs the Vulkan SDK (glslc)
CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89" pip install .   # untested
# keep separate build trees per variant:
pip wheel . -C build-dir=build/vulkan -C cmake.define.GGML_VULKAN=ON
```

### CUDA

Building needs the CUDA 12 toolkit (nvcc), not a GPU or driver:

```bash
CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86-real" \
    pip wheel . -C build-dir=build/cuda
```

- Set `CMAKE_CUDA_ARCHITECTURES` explicitly; `native` needs a GPU. Each
  architecture adds substantial nvcc time. Release wheels use
  `75-real;80-real;86-real;89-real;90-real;120` (Turing to Blackwell, plus
  PTX for newer GPUs).
- The wheel is self-contained apart from the NVIDIA driver: the repair step
  bundles the CUDA runtime, cuBLAS and cuBLASLt (about 600 MB), which are
  NVIDIA redistributables under the CUDA Toolkit EULA (shipped in
  `nemo_speech/licenses/third_party/cuda/`). Repair with
  `auditwheel repair --exclude libcuda.so.1` or
  `delvewheel repair --analyze-existing --ignore-existing --exclude nvcuda.dll`
  so the driver itself is never bundled. Without a driver the import fails
  with a message saying so.
- Upstream's drop-in cuBLAS shim (`NEMO_SPEECH_CUBLAS_SHIM`) does not cover
  the pinned ggml-cuda (it lacks `cublasSetWorkspace_v2` and
  `cublasSgemmBatched`); the build refuses it until upstream catches up.
- Windows: nvcc cannot use a `%TEMP%` path with non-ASCII characters (for
  example a Japanese user name); point `TEMP`/`TMP` at an ASCII directory for
  the build.

CI builds the CUDA wheels on tags and manual runs (`wheels-cuda` job).

Wheel defaults: ASR, diarization, TTS and NMT on; CLI and microphone capture
off; `GGML_NATIVE=OFF` (portable AVX2 baseline on x86-64); on macOS Metal on
and OpenMP off. The Japanese/Mandarin TTS tokenizers are off (see Known issues).

## Usage

```python
from nemo_speech import Diarizer, Recognizer, Synthesizer, Translator, load_wav

# ASR. from_pretrained() downloads from the catalog; a local path works too.
with Recognizer.from_pretrained("nemotron-3.5", gpu=-1) as asr:
    result = asr.transcribe_file("speech.wav", word_timestamps=True)
    print(result.text, [(w.text, w.start, w.end) for w in result.words])

    samples, rate = load_wav("speech.wav")      # or any mono numpy array
    with asr.stream() as stream:
        for i in range(0, samples.size, rate // 2):
            for r in stream.push(samples[i : i + rate // 2], rate):
                print("final" if r.is_final else "partial", r.text)
        for r in stream.finish():
            print("final", r.text)

# Speaker diarization ("who spoke when").
with Diarizer.from_pretrained(gpu=-1) as diar:
    for seg in diar.diarize_file("meeting.wav").segments:
        print(f"{seg.start:6.2f}-{seg.end:6.2f} speaker {seg.speaker}")

# TTS: model, codec and tokenizer assets come from the catalog together.
with Synthesizer.from_pretrained("magpie") as tts:
    print(tts.speakers)
    tts.synthesize("Hello world.", voice="Sofia").save("hello.wav")
    for chunk in tts.stream("Streaming playback, chunk by chunk."):
        ...  # int16 numpy arrays at tts.sample_rate

# NMT: Riva-Translate is not in the catalog; convert it or use a GGUF.
with Translator("riva-translate-4b-instruct-v2.q8_0.gguf", gpu=-1) as nmt:
    print(nmt.translate(["Good morning.", "Thank you."], "en", "ja"))
```

Notes:

- `gpu=None` uses the library default (device 0 when the build has a GPU
  backend); `-1` forces the CPU. TTS has no device option in the C ABI.
- `Recognizer`, `Diarizer` (separate streams) and `Translator` can be shared
  across threads; ctypes releases the GIL during native calls. Calls on one
  `Synthesizer` are serialized.
- `nemo_speech.capi.{asr,diar,nmt,tts}` expose the C ABI directly.
- Building without a component (e.g. `-DNEMO_SPEECH_BUILD_NMT=OFF`) keeps
  `import nemo_speech` working; only that class raises `ImportError`.

### Models

`nemo_speech.models` reads upstream's `models/index.json` and downloads into
the same cache as the `nemo-speech` CLI, verifying size and SHA-256:
`NEMO_SPEECH_MODEL_DIR`, else `%LOCALAPPDATA%\NeMoSpeech\models` (Windows),
`~/Library/Caches/NeMoSpeech/models` (macOS), `~/.cache/nemo-speech/models`
(Linux). The TTS tokenizer is fetched with HTTP range requests from the
`.nemo` archive (about 31 MB of a 1.4 GB file), as the CLI does.

```python
from nemo_speech import models
models.download("magpie", companions=True)   # {"tts": ..., "tokenizer": ..., "codec": ...}
```

### Performance (Intel Iris Xe laptop, Windows)

| | CPU wheel, `gpu=-1` | Vulkan wheel, `gpu=0` |
|---|---|---|
| ASR offline, 11 s clip (nemotron-3.5 q8_0) | 4.1 s (RTF 0.37) | 8.5 s (RTF 0.78) |
| TTS (MagpieTTS) | RTF 1.51 | RTF 1.51 (runs on the CPU) |
| NMT, one sentence (Riva-Translate 4B Q4_K_M) | 4.8 s | 1.7 s |

On this integrated GPU, Vulkan helps NMT but slows ASR down: upstream's fused
attention operations are CUDA-only, so the graph is split between the GPU and
the CPU. With the Vulkan wheel, pass `gpu=-1` to the recognizer.

### Runtime requirements

- Windows: nothing beyond the wheel; the MSVC/OpenMP runtime is bundled in
  `nemo_speech.libs` by delvewheel. Vulkan wheels need a GPU driver providing
  `vulkan-1.dll`.
- `NEMO_SPEECH_LIB_PATH=<dir>` loads the libraries from another directory, for
  example a local NeMo-Speech.cpp build.

## Development

```bash
pip install .[test]
pytest
```

Run the `pytest` executable rather than `python -m pytest` from the repository
root: the latter puts the source tree first on `sys.path`, and the source
`nemo_speech/` has no `lib/` directory.

Model-backed tests never download. They use `NEMO_SPEECH_TEST_ASR_MODEL`,
`NEMO_SPEECH_TEST_DIAR_MODEL` and `NEMO_SPEECH_TEST_NMT_MODEL`, files in
`models/`, or models already in the cache, and skip otherwise.
`NEMO_SPEECH_TEST_GPU=0` runs them on a GPU. `tests/test_capi_abi.py` parses
the vendored headers and fails when a struct field or exported function is
missing from `nemo_speech/capi`; run it after moving the submodule.

### CI and releases

Every wheel is built from the sdist with cibuildwheel and repaired with
auditwheel/delvewheel/delocate. Builds are tiered by cost:

| Workflow | When | Builds |
|---|---|---|
| `wheels.yml` | every push and PR | CPU wheels (Linux x86_64/aarch64, Windows x64, macOS arm64, all tested) and the Windows Vulkan wheel, about 20 minutes |
| `cuda-smoke.yml` | pushes and PRs that touch `vendor/`, `CMakeLists.txt`, `cmake/`, `patches/`, `pyproject.toml`, the loader or the CUDA workflows | CUDA wheels for one architecture (sm_86), Linux tested against the driver stub; nothing uploaded |
| `wheels.yml` `wheels-cuda` | tags `v*` and manual runs | CUDA wheels for all release architectures; a manual run can pick `target: cuda`, `cuda-windows` or `cuda-linux` |

Python-only changes cannot break the CUDA build (the bindings use ctypes),
which is why the CUDA tiers watch only the native inputs. `sdist.yml` and
`cuda.yml` are the reusable pieces shared by both workflows.

The cibuildwheel settings live in `pyproject.toml`; the Linux and Windows
parts run locally too:

```bash
python -m build --sdist -o dist/sdist
cibuildwheel dist/sdist/*.tar.gz --platform linux --archs x86_64   # needs Docker
cibuildwheel dist/sdist/*.tar.gz --platform windows
```

GPU wheels share the distribution name with the CPU wheel, so like
llama-cpp-python they belong on a separate package index. Publishing is not
configured yet.

## Known issues

Upstream (worth reporting to NeMo-Speech.cpp):

- **llama.cpp patching on Windows.** Upstream normalizes each patch with CMake
  `file(WRITE)`, which writes CRLF on Windows; with `core.autocrlf=input` or
  `false` the patches then fail to apply. This build applies the series itself
  and passes the result as `NEMO_SPEECH_LLAMA_CPP_SOURCE_DIR`.
- **TTS cancellation status.** Returning `false` from the PCM callback is
  reported as `NEMO_SPEECH_TTS_ERROR_RUNTIME`, not `..._CANCELLED`, on the
  streaming codec path. The bindings treat any failure after a requested
  cancellation as the cancellation.
- **Diarization example.** `examples/diarize_file.cpp` zero-initializes
  `left_context_frames`, which is a valid explicit value (0) rather than
  "keep the preset"; the bindings pass -1.
- **Non-relocatable TTS data.** The Japanese/Mandarin tokenizers compile
  absolute data paths into the library, so they cannot ship in a wheel.
- **cuBLAS shim out of date.** Since the llama.cpp update in a5f19be, ggml-cuda
  calls `cublasSetWorkspace_v2` and `cublasSgemmBatched`, which
  `kernels/cublas_shim.cu` does not provide, so shim-based CUDA builds fail to
  load.

This package:

- **llama.cpp and manylinux_2_28.** llama.cpp's sampler calls
  `std::random_device::entropy()`, which needs `GLIBCXX_3.4.25`; the
  manylinux_2_28 policy allows up to 3.4.24, so auditwheel refuses the wheel.
  `patches/llama.cpp/` skips that check with libstdc++, whose random_device is
  always a true RNG on the supported targets. (The alternative, a
  manylinux_2_31 tag, would exclude RHEL/Alma 8.)
- **Unmangled MSVC runtime.** delvewheel's `--with-mangle` fails on this
  layout (1.10-1.13), so `msvcp140.dll` is bundled under its own name. The
  known crash with an older, already-loaded runtime is avoided by building
  with `_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR`. The bundled `msvcp140.dll` comes
  from the build machine (delvewheel warns when it is newer than Python's own
  `vcruntime140.dll`, which the process uses regardless).
- **Non-ASCII paths on Windows (handled).** Paths are passed to the C ABI as
  UTF-8. Most components open them with `ggml_fopen` (UTF-8 aware), but the
  MagpieTTS and NanoCodec models, the TTS tokenizer assets and the ASR
  profanity list use narrow `fopen` / `std::ifstream` / `std::filesystem`,
  which the C runtime reads in the ANSI code page (932 on a default Japanese
  Windows). A model cache under a Japanese user name would then fail to load
  (the profanity list would silently be empty). The high-level classes wrap
  model creation in `nemo_speech.capi.utf8_file_paths()`, which gives the
  calling thread a UTF-8 C locale for that call only; low-level
  `nemo_speech.capi` users should do the same. Systems with the "Beta: Use
  Unicode UTF-8" setting were never affected. 8.3 short names are no
  alternative: a name of up to eight bytes in the OEM code page, such as a
  four-kanji user name, has no separate short name.
- **Logging.** The libraries log to stderr; the C ABI has no switch for it.

## License

Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Parts are
adapted from NeMo-Speech.cpp (Apache-2.0). Wheels bundle NeMo-Speech.cpp,
ggml/llama.cpp, SentencePiece and runtime libraries, whose licenses ship in
`nemo_speech/licenses/`.

Models are not part of this project. `nemo_speech.models` downloads them from
their publishers on request, and each model is governed by its own license
(listed as `license_url` in the catalog), which you are responsible for
reviewing.
