"""Minimal WAV input without third-party dependencies.

Reads integer PCM WAV (8/16/24/32-bit), including WAVE_FORMAT_EXTENSIBLE
files, which many tools write for 24-bit or multi-channel audio. For float
WAV, FLAC, MP3 and so on, use a library such as ``soundfile`` and pass the
samples to the recognizer directly.
"""

from __future__ import annotations

import struct
import wave
from typing import Tuple

import numpy as np

from ._common import PathLike, pcm_to_f32

_WAVE_FORMAT_PCM = 0x0001
_WAVE_FORMAT_IEEE_FLOAT = 0x0003
_WAVE_FORMAT_EXTENSIBLE = 0xFFFE
# KSDATAFORMAT_SUBTYPE_* GUIDs are the format tag followed by these 14 bytes.
_SUBFORMAT_GUID_TAIL = bytes.fromhex("000000001000800000aa00389b71")
# Writers that stream WAV to a pipe leave the data size at its maximum.
_UNKNOWN_SIZE = 0xFFFFFFFF


def _decode_frames(raw: bytes, width: int) -> np.ndarray:
    if width == 1:
        return pcm_to_f32(np.frombuffer(raw, dtype=np.uint8))
    if width == 2:
        return pcm_to_f32(np.frombuffer(raw, dtype="<i2"))
    if width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v & 0x800000, v - 0x1000000, v)
        return v.astype(np.float32) / float(0x800000)
    if width == 4:
        return pcm_to_f32(np.frombuffer(raw, dtype="<i4"))
    raise ValueError(f"unsupported WAV sample width: {width} bytes")


def _parse_fmt(path: PathLike, fmt: bytes) -> Tuple[int, int, int]:
    """``(channels, sample_rate, bytes per sample)`` of an integer PCM ``fmt `` chunk."""
    if len(fmt) < 16:
        raise ValueError(f"{path}: truncated WAV fmt chunk")
    tag, channels, rate, _, block_align, _ = struct.unpack("<HHIIHH", fmt[:16])
    if tag == _WAVE_FORMAT_EXTENSIBLE:
        if len(fmt) < 40 or fmt[26:40] != _SUBFORMAT_GUID_TAIL:
            raise ValueError(f"{path}: unsupported WAVE_FORMAT_EXTENSIBLE sub-format")
        (tag,) = struct.unpack("<H", fmt[24:26])
    if tag == _WAVE_FORMAT_IEEE_FLOAT:
        raise ValueError(
            f"{path}: float WAV is not supported; read it with a library such as "
            "soundfile and pass the samples instead"
        )
    if tag != _WAVE_FORMAT_PCM:
        raise ValueError(f"{path}: unsupported WAV format 0x{tag:04x} (integer PCM expected)")
    if channels == 0 or block_align % channels:
        raise ValueError(f"{path}: invalid WAV fmt chunk ({channels} channels, block {block_align})")
    # Samples are left-justified in their container, so a 24-bit sample in a
    # 32-bit container decodes as 32-bit.
    return channels, rate, block_align // channels


def _read_pcm(path: PathLike) -> Tuple[bytes, int, int, int]:
    """``(frames, channels, sample_rate, bytes per sample)`` of an integer PCM WAV."""
    with open(path, "rb") as f:
        header = f.read(12)
        if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise ValueError(f"{path}: not a RIFF/WAVE file")
        layout = None
        while True:
            chunk = f.read(8)
            if len(chunk) < 8:
                raise ValueError(f"{path}: no WAV data chunk")
            chunk_id, size = chunk[:4], struct.unpack("<I", chunk[4:])[0]
            if chunk_id == b"fmt ":
                layout = _parse_fmt(path, f.read(size))
            elif chunk_id == b"data":
                if layout is None:
                    raise ValueError(f"{path}: WAV data chunk before the fmt chunk")
                raw = f.read() if size == _UNKNOWN_SIZE else f.read(size)
                channels, rate, width = layout
                # A truncated file may end inside a frame.
                raw = raw[: len(raw) - len(raw) % (channels * width)]
                return raw, channels, rate, width
            else:
                f.seek(size, 1)
            if size % 2:
                f.seek(1, 1)  # chunks are padded to an even size


def load_wav(path: PathLike) -> Tuple[np.ndarray, int]:
    """Read ``path`` as ``(mono float32 samples, sample_rate)``.

    Multi-channel files are downmixed by averaging the channels.
    """
    raw, channels, rate, width = _read_pcm(path)
    samples = _decode_frames(raw, width)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(samples, dtype=np.float32), rate


def save_wav(path: PathLike, samples: "np.typing.ArrayLike", sample_rate: int) -> None:
    """Write mono samples as 16-bit PCM WAV (floats are clipped to [-1, 1])."""
    a = np.asarray(samples)
    if a.dtype != np.int16:
        a = np.round(np.clip(pcm_to_f32(a), -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(a.astype("<i2").tobytes())
