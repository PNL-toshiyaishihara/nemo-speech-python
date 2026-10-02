"""Model catalog and downloads, compatible with the ``nemo-speech`` CLI.

The catalog is upstream's ``models/index.json`` (bundled as
``model-index.json``). Files go to the same cache as the CLI, so models
pulled by either are shared:

- ``NEMO_SPEECH_MODEL_DIR`` if set, else
- Windows: ``%LOCALAPPDATA%\\NeMoSpeech\\models``
- macOS: ``~/Library/Caches/NeMoSpeech/models``
- Linux: ``$XDG_CACHE_HOME/nemo-speech/models`` or ``~/.cache/nemo-speech/models``

laid out as ``<root>/<org>/<name>/<revision>/<file or directory>``.
``NEMO_SPEECH_HF_BASE_URL`` overrides the Hugging Face endpoint (HTTPS only,
HTTP for loopback hosts).

Ported to Python from NeMo-Speech.cpp app/model_store.cpp (Copyright (c) 2026
NVIDIA CORPORATION & AFFILIATES, Apache-2.0). Downloaded models are governed by
their own licenses (see each catalog entry's ``license_url``).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import pathlib
import shutil
import sys
import tarfile
import urllib.parse
import urllib.request
from typing import Any, BinaryIO, Dict, List, Optional

__all__ = ["cache_root", "default", "download", "find", "index", "list_models"]

_INDEX_PATH = pathlib.Path(__file__).resolve().parent / "model-index.json"
_CHUNK = 1 << 20


def index() -> Dict[str, Any]:
    """The parsed model catalog."""
    with open(_INDEX_PATH, encoding="utf-8") as f:
        return json.load(f)


def list_models() -> List[Dict[str, Any]]:
    return list(index()["models"])


def default(role: str) -> str:
    """Repository of the catalog's default model for ``role`` (e.g. "asr")."""
    defaults = index().get("defaults", {})
    if role not in defaults:
        raise KeyError(f"no default model for role {role!r}; known: {sorted(defaults)}")
    return defaults[role]


def find(name: str) -> Dict[str, Any]:
    """Look up a model by repository ("nvidia/...") or alias (case-insensitive)."""
    key = name.lower()
    for model in index()["models"]:
        if model["repo"].lower() == key or key in (a.lower() for a in model.get("aliases", [])):
            return model
    raise KeyError(f"unknown model {name!r}")


def cache_root() -> pathlib.Path:
    override = os.environ.get("NEMO_SPEECH_MODEL_DIR")
    if override:
        return pathlib.Path(override)
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        if local:
            return pathlib.Path(local) / "NeMoSpeech" / "models"
    elif sys.platform == "darwin":
        return pathlib.Path.home() / "Library" / "Caches" / "NeMoSpeech" / "models"
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        if xdg:
            return pathlib.Path(xdg) / "nemo-speech" / "models"
        return pathlib.Path.home() / ".cache" / "nemo-speech" / "models"
    raise RuntimeError("cannot determine the model cache directory; set NEMO_SPEECH_MODEL_DIR")


def _base_url() -> str:
    url = os.environ.get("NEMO_SPEECH_HF_BASE_URL") or "https://huggingface.co"
    url = url.rstrip("/")
    parsed = urllib.parse.urlsplit(url)
    loopback = parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost", "::1")
    if parsed.scheme != "https" and not loopback:
        raise ValueError(
            "NEMO_SPEECH_HF_BASE_URL must use HTTPS (HTTP is allowed only for loopback tests)"
        )
    return url


def _model_dir(model: Dict[str, Any]) -> pathlib.Path:
    org, name = model["repo"].split("/", 1)
    return cache_root() / org / name / model["revision"]


def _url(model: Dict[str, Any], artifact: Dict[str, Any]) -> str:
    revision = artifact.get("revision") or model["revision"]
    return f"{_base_url()}/{model['repo']}/resolve/{revision}/{artifact['filename']}?download=true"


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _valid_file(path: pathlib.Path, size: int, sha256: Optional[str]) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    return sha256 is None or _sha256(path) == sha256


class _Progress:
    def __init__(self, label: str, total: int, enabled: bool) -> None:
        self.label, self.total, self.enabled, self.done, self._shown = label, total, enabled, 0, -1

    def update(self, n: int) -> None:
        self.done += n
        if not self.enabled or not self.total:
            return
        percent = self.done * 100 // self.total
        if percent != self._shown:
            self._shown = percent
            sys.stderr.write(f"\r[model] {self.label}: {percent:3d}%")
            if self.done >= self.total:
                sys.stderr.write("\n")
            sys.stderr.flush()


def _fetch(url: str, out: BinaryIO, progress: _Progress, byte_range: Optional[str] = None) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "nemo-speech-python"})
    if byte_range:
        request.add_header("Range", f"bytes={byte_range}")
    with urllib.request.urlopen(request) as response:
        if byte_range and response.status != 206:
            raise RuntimeError(f"server ignored the byte range request for {url}")
        for block in iter(lambda: response.read(_CHUNK), b""):
            out.write(block)
            progress.update(len(block))


def _safe_member_path(destination: pathlib.Path, name: str) -> pathlib.Path:
    relative = pathlib.PurePosixPath(name)
    if not name or relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"unsafe path in tokenizer artifact: {name!r}")
    return destination.joinpath(*relative.parts)


def _extract_segment(data: bytes, destination: pathlib.Path, stop_before: Optional[str]) -> None:
    """Extract regular files from one TAR segment of a ranged archive."""
    reached_stop = False
    with tarfile.open(fileobj=io.BytesIO(data), mode="r|") as archive:
        for member in archive:
            if stop_before and pathlib.PurePosixPath(member.name).name == stop_before:
                reached_stop = True
                break
            target = _safe_member_path(destination, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isreg():
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                assert source is not None
                with open(target, "wb") as f:
                    shutil.copyfileobj(source, f)
            else:
                raise RuntimeError(f"unsupported TAR entry in tokenizer artifact: {member.name!r}")
    if stop_before and not reached_stop:
        raise RuntimeError("tokenizer archive range did not reach the expected stop member")


def _materialize(model: Dict[str, Any], artifact: Dict[str, Any], progress: bool) -> pathlib.Path:
    directory = _model_dir(model)
    directory.mkdir(parents=True, exist_ok=True)
    kind = artifact["type"]

    if kind == "file":
        destination = directory / artifact["filename"]
        if _valid_file(destination, artifact["size"], artifact.get("sha256")):
            return destination
        partial = destination.with_name(destination.name + ".partial")
        with open(partial, "wb") as out:
            _fetch(_url(model, artifact), out, _Progress(artifact["filename"], artifact["size"], progress))
        if not _valid_file(partial, artifact["size"], artifact.get("sha256")):
            partial.unlink()
            raise RuntimeError(f"downloaded {artifact['filename']} failed verification")
        os.replace(partial, destination)
        return destination

    if kind == "tar-ranges":
        destination = directory / artifact["directory"]
        members = artifact["members"]
        if destination.is_dir() and all(
            _valid_file(destination / m["name"], m["size"], m.get("sha256")) for m in members
        ):
            return destination
        bar = _Progress(artifact["directory"], artifact["size"], progress)
        segments = []
        for r in artifact["ranges"]:
            buffer = io.BytesIO()
            _fetch(_url(model, artifact), buffer, bar, f"{r['start']}-{r['end']}")
            if buffer.tell() != r["end"] - r["start"] + 1:
                raise RuntimeError("truncated tokenizer archive range")
            segments.append((buffer.getvalue(), r.get("stop_before")))
        extracting = directory / (artifact["directory"] + ".extracting")
        shutil.rmtree(extracting, ignore_errors=True)
        extracting.mkdir()
        for data, stop_before in segments:
            _extract_segment(data, extracting, stop_before)
        for m in members:
            if not _valid_file(extracting / m["name"], m["size"], m.get("sha256")):
                raise RuntimeError(f"tokenizer artifact member failed verification: {m['name']}")
        shutil.rmtree(destination, ignore_errors=True)
        os.replace(extracting, destination)
        return destination

    raise NotImplementedError(f"artifact type {kind!r} is not supported")


def download(
    name: str, *, companions: bool = False, progress: bool = True
) -> Dict[str, pathlib.Path]:
    """Ensure every artifact of ``name`` is cached; return ``{role: path}``.

    Cached files are verified (size and SHA-256) and reused. With
    ``companions=True`` the model's companions (e.g. the TTS codec) are
    downloaded too and merged into the result.
    """
    model = find(name)
    paths: Dict[str, pathlib.Path] = {}
    if companions:
        for companion in model.get("companions", []):
            paths.update(download(companion, progress=progress))
    for artifact in model["artifacts"]:
        paths[artifact["role"]] = _materialize(model, artifact, progress)
    return paths
