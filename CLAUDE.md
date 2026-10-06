# Notes for Claude

Read [CONTRIBUTING.md](CONTRIBUTING.md) (principles, setup, Windows pitfalls,
test models) and [RELEASING.md](RELEASING.md) (versioning, labels, release
steps) before changing anything. The rules that are easiest to get wrong:

- Never modify `vendor/` (submodules). Work around upstream problems in this
  repository and record them as `upstream` issues; do not propose changes
  upstream.
- Wheels must work on their own: bundle native libraries, depend only on
  Python wheels.
- All changes go through branches and PRs into `main`; label every PR.
- Verify with the installed wheel: build with `pip wheel`, install it, run
  the `pytest` executable (not `python -m pytest`). Run long builds in the
  background, one build per build directory.
- This development machine has no NVIDIA GPU and uses the UTF-8 ANSI code
  page; say so instead of claiming CUDA or non-ASCII-path behavior was
  verified locally.
- Commit with the identity configured in this clone; do not change it.

Language: code, comments and repository documents are in English. Issues
and discussion with the maintainer are in Japanese; sign issue text written
by Claude as such.
