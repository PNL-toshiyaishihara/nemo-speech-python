# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately through
[GitHub's private vulnerability reporting](https://github.com/PNL-toshiyaishihara/nemo-speech-python/security/advisories/new),
not in a public issue. Include the output of `nemo_speech.build_info()` (or the
name of the wheel you installed), your platform, and how to reproduce the
problem.

The project has a single maintainer, so reports are handled on a best-effort
basis without a guaranteed response time. Once a fix is released, the report
is published as a security advisory that credits the reporter, unless they
prefer not to be named.

## Supported versions

Only the latest release receives fixes. Published releases are immutable, so a
fix always ships as a new release.

## Scope

In scope: the Python code in this repository, the build and packaging, and the
released wheels as shipped, including the native libraries they bundle.

The native code comes from NeMo-Speech.cpp, llama.cpp/ggml, SentencePiece,
Open JTalk and cppjieba, pinned as submodules. A vulnerability in one of them
belongs to that project, so please report it there. If a released wheel is
affected, report it here as well: the fix on this side is a new release with
a fixed pin or a workaround.

Model files are input to native code: the bundled libraries parse them without
any sandbox. Load models only from sources you trust. `nemo_speech.models`
checks the size and SHA-256 of every file it downloads against upstream's
catalog.
