# Releasing

Releases are GitHub Releases built by `.github/workflows/release.yml` from a
tag. PyPI is not used yet (see the end).

## Versioning

- The package has its **own** version, independent of NeMo-Speech.cpp's. The
  upstream revision of each release is recorded in its notes and in
  `nemo_speech.build_info()`.
- Semantic versioning in PEP 440 form: `0.2.0`, `0.2.1`, pre-releases
  `0.2.0rc1` (not `0.2.0-rc.1`). While on `0.x`:
  - **minor** (`0.3.0`): new features, behavior changes, NeMo-Speech.cpp
    updates, anything labelled `breaking`;
  - **patch** (`0.2.1`): fixes only.
- `__version__` in `nemo_speech/__init__.py` is the single source of the
  version. A release tag is exactly `v` + that version, and the release
  workflow refuses anything else. Between releases `main` carries the next
  version with `.dev0` (e.g. `0.3.0.dev0`).

## Variants

One release contains these wheels; all are `py3-none`, so one file serves
Python 3.10 to 3.14 (each is tested on all of them, except where noted).

| Variant | Version label | `NEMO_SPEECH_VARIANT` | Platforms |
|---|---|---|---|
| CPU (Metal on macOS) | none | unset | Linux x86_64, Linux aarch64, Windows x64, macOS arm64 |
| Vulkan | `+vulkan` | `vulkan` | Windows x64 |
| CUDA 12.8 | `+cu128` | `cu128` | Linux x86_64, Windows x64 (not testable on CI runners) |
| CUDA 13.0 | `+cu130` | `cu130` | Linux x86_64, Windows x64 (not testable on CI runners) |

The local version label keeps the file names apart. `NEMO_SPEECH_VARIANT`
enables the backend and sets the label (`pyproject.toml` overrides). The build
refuses unknown values, a GPU backend without its variant, and CUDA toolkits
that do not match the label. A new variant needs an override in
`pyproject.toml` and an entry in `NSP_KNOWN_VARIANTS` in `CMakeLists.txt`.

A release therefore has 11 assets: 9 wheels, the sdist and `SHA256SUMS`.

## Branches, PRs and labels

- `main` is always releasable. All changes go through PRs (squash merge), so
  CI runs before merging and each PR title becomes one release-notes entry.
- Label every PR; `.github/release.yml` turns the labels into the sections of
  the generated notes:

  | Label | Section |
  |---|---|
  | `breaking` | Breaking changes |
  | `enhancement` | Features |
  | `bug` | Fixes |
  | `dependencies` | Vendored dependencies (submodule updates) |
  | `ci` | Packaging and CI |
  | `documentation` | Documentation |
  | `skip-changelog` | left out (version bumps, release chores) |

- Maintenance branches (`release/0.2`) only if an old line ever needs a fix.

## Making a release

1. **Version PR.** Set `__version__` to the release version (`0.2.0`, or
   `0.2.0rc1` to rehearse), label it `skip-changelog`, merge it.
2. **Tag** the merge commit on `main` and push the tag:

   ```bash
   git switch main && git pull
   git tag -a v0.2.0 -m "nemo-speech-python 0.2.0"
   git push origin v0.2.0
   ```

3. **Wait** for `release.yml` (about two hours, dominated by the CUDA
   builds). It refuses a tag that is not on `main` or does not match
   `__version__`, builds and tests every variant, checks that every expected
   wheel and the sdist are there, then creates a **draft** release with the
   assets, `SHA256SUMS` and generated notes. Tags with `a`/`b`/`rc`/`dev`
   become pre-releases.
4. **Review the draft**: the change list, the vendored-source table, that all
   11 assets are there. Edit the text if needed, then publish it.
5. **Next development version.** A PR setting `__version__` to the next
   `.dev0`, labelled `skip-changelog`: after `0.2.0`, `0.3.0.dev0`. After a
   pre-release, the next pre-release's `.dev0` (after `0.2.0rc1`,
   `0.2.0rc2.dev0`), because PEP 440 sorts `0.2.0.dev0` *before*
   `0.2.0rc1`. The final release's version PR then sets `0.2.0`.

If the release workflow fails before anything is published, fix the cause in
a PR, delete the tag (`git push --delete origin v0.2.0`, `git tag -d v0.2.0`)
and tag again. The next run deletes any draft an earlier run left for the
same tag, and it refuses to touch a published release. Never move or reuse
the tag of a published release; release a new patch version instead.

## Updating NeMo-Speech.cpp

- Prefer upstream release tags over arbitrary commits.
- One PR per update, labelled `dependencies`. Changes under `vendor/` trigger
  `cuda-smoke.yml` (single-architecture CUDA builds for both toolkits) in
  addition to the regular CI.
- Check the `upstream` issues: workarounds for bugs fixed upstream can go.

## What the notes contain

`scripts/release_notes.py` writes the notes from the built wheels (their
`nemo_speech/_build_info.json`) and git:

- the generated change list since the previous release (for a final
  release, the previous final release; the first final release lists
  everything, combined from the ranges between its pre-releases, because
  GitHub would otherwise start at the last pre-release);
- the vendored sources (NeMo-Speech.cpp, llama.cpp, SentencePiece) with their
  commits, and a compare link for NeMo-Speech.cpp against the previous
  release's pin;
- every wheel with its variant, platform and requirements;
- install instructions and the `upstream` known-issues link.

## PyPI (later)

PyPI rejects local version labels, so only the CPU wheels and the sdist can go
there; GPU variants stay on GitHub Releases (or a per-variant package index,
as llama-cpp-python does). Publishing would add a job to `release.yml` using
PyPI trusted publishing. Decide the distribution name first: `nemo-speech`
is free on PyPI but contains NVIDIA's "NeMo" name.
