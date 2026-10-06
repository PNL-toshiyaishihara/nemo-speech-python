# Contributing

How this project is built and changed. Releases and versioning are in
[RELEASING.md](RELEASING.md); the build and CI overview is in the
[README](README.md).

## Principles

1. **A wheel works on its own.** `pip install` of one wheel must be enough.
   Runtime dependencies may only be other Python wheels (today: numpy).
   Native libraries are bundled by the repair step (auditwheel, delvewheel,
   delocate) even when that makes a wheel large: the CUDA wheels carry the CUDA
   runtime, cuBLAS and cuBLASLt (about 700 MB). The only host requirement is
   the GPU driver. Never ask users to install a toolkit or edit `PATH`.
2. **Upstream stays untouched.** `vendor/NeMo-Speech.cpp` is a pinned
   submodule that is neither modified nor patched from here, and this
   project does not send changes upstream. Problems are solved inside this
   repository: the CMake superbuild, the llama.cpp patch series in
   `patches/llama.cpp/`, or the Python layer.
3. **Record upstream defects as issues** labelled `upstream`: the symptom,
   a permalink to the upstream code, the workaround here, a possible upstream
   fix, and a checklist. When an update of the submodule fixes one, remove
   the workaround and close the issue.
4. **Only the stable C ABI.** The bindings use `include/nemo_speech/*.h`
   through ctypes; `nemo_speech/capi/` mirrors the headers one-to-one, and
   `tests/test_capi_abi.py` fails when they drift apart.
5. **Own version.** The package version is independent of NeMo-Speech.cpp's
   (see RELEASING.md).

## Workflow

- Every change goes through a PR into `main`, squash-merged; the PR title
  becomes the release-notes entry. Label each PR (the labels are listed in
  RELEASING.md) and fill in the template.
- Stacked PRs: before deleting the branch of a merged PR, retarget every PR
  based on it to `main`. Deleting the base branch closes those PRs instead of
  retargeting them.
- Updating NeMo-Speech.cpp: one PR per update, preferably to an upstream
  release tag, labelled `dependencies`.

## Development setup

```bash
git submodule update --init vendor/NeMo-Speech.cpp vendor/sentencepiece
git -C vendor/NeMo-Speech.cpp submodule update --init --depth 1 llama.cpp
git -C vendor/NeMo-Speech.cpp submodule update --init --recursive --depth 1 third_party/open_jtalk third_party/cppjieba
python -m venv .venv && .venv/Scripts/python -m pip install .[test]   # bin/ on Linux and macOS
.venv/Scripts/pytest
```

- Run the `pytest` executable, not `python -m pytest` from the repository
  root: the latter imports the source tree, which has no `nemo_speech/lib/`.
- Build trees persist under `build/<wheel tag>/`, so rebuilds are
  incremental. Use one build tree per variant
  (`-C build-dir=build/vulkan`), and never run two builds in the same tree at
  once: they corrupt each other.
- Variants: `NEMO_SPEECH_VARIANT=vulkan` (or `cu128`, `cu130`) enables the
  backend and the version label. CUDA additionally needs
  `CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=86-real"` (any explicit list; no GPU
  is needed to build).
- Linux wheels can be built locally with Docker:
  `cibuildwheel <sdist> --platform linux --archs x86_64`.

### Windows

- Visual Studio 2022 (or its Build Tools) with the C++ workload.
- Keep paths short. The nested Vulkan shader generator exceeds MAX_PATH from
  deep directories; build with `-C build-dir=C:/b/...` if in doubt.
- nvcc cannot use a `%TEMP%` path with non-ASCII characters (a Japanese user
  name, for example): set `TEMP` and `TMP` to an ASCII directory for CUDA
  builds.
- A machine with the "Beta: Use Unicode UTF-8 for worldwide language support"
  setting (ANSI code page 65001) hides bugs with non-ASCII paths, because the
  narrow C runtime APIs then accept UTF-8. Such bugs only show on the default
  code pages (932 on Japanese Windows, 1252 on the GitHub runners), so rely on
  CI for them.

## Testing

Model-backed tests skip unless the model is available. They look for:

| Test | Model | Source |
|---|---|---|
| ASR | `NEMO_SPEECH_TEST_ASR_MODEL`, `models/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf`, or the cache | `nemo_speech.models.download("nemotron-3.5")` |
| Diarization | `NEMO_SPEECH_TEST_DIAR_MODEL` or the cache | `nemo_speech.models.download("nemotron-diar")` |
| TTS | the cache | `nemo_speech.models.download("magpie", companions=True)` |
| NMT | `NEMO_SPEECH_TEST_NMT_MODEL` or `models/Riva-Translate-4B-Instruct-v2-Q4_K_M.gguf` | not in the catalog: convert `nvidia/Riva-Translate-4B-Instruct-v2` with upstream's `convert_model.py`, or use a community GGUF (development used `shivnathtathe/nvidia_Riva-Translate-4B-Instruct-v2-GGUF`, Q4_K_M) |

`NEMO_SPEECH_TEST_GPU=0` runs the model tests on GPU device 0.

Known gaps in what has been verified:

- No CUDA wheel has run on an NVIDIA GPU yet; CI builds and repairs them,
  and tests the Linux ones on the CPU through the driver stub.
- Non-ASCII paths are verified at the C runtime level on CI, not with real
  models on a code-page-932 system.

These are tracked as issues.
