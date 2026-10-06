import struct

import numpy as np
import pytest

from nemo_speech import load_wav, save_wav
from nemo_speech._common import as_mono_f32

GUID_TAIL = bytes.fromhex("000000001000800000aa00389b71")


def _write_wav(path, frames, *, channels=1, rate=16000, container_bits=16, valid_bits=None,
               extensible=False, format_tag=1, before_data=b"", data_size=None):
    """Write a WAV file by hand, as tools other than Python's wave module do."""
    block = channels * container_bits // 8
    fmt = struct.pack("<HHIIHH", 0xFFFE if extensible else format_tag, channels, rate,
                      rate * block, block, container_bits)
    if extensible:
        fmt += struct.pack("<HHI", 22, valid_bits or container_bits, 0)
        fmt += struct.pack("<H", format_tag) + GUID_TAIL
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + before_data
    body += b"data" + struct.pack("<I", len(frames) if data_size is None else data_size) + frames
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
    return path


def _int24(values):
    return b"".join(int(v).to_bytes(3, "little", signed=True) for v in values)


def test_int16_is_scaled_to_unit_range():
    out = as_mono_f32(np.array([-32768, 0, 16384], dtype=np.int16))
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [-1.0, 0.0, 0.5])


def test_uint8_is_centered():
    np.testing.assert_allclose(as_mono_f32(np.array([0, 128], dtype=np.uint8)), [-1.0, 0.0])


def test_column_vector_is_accepted():
    assert as_mono_f32(np.zeros((10, 1), dtype=np.float32)).shape == (10,)


def test_multichannel_is_rejected():
    with pytest.raises(ValueError, match="mono"):
        as_mono_f32(np.zeros((10, 2), dtype=np.float32))


def test_load_wav(jfk_wav):
    samples, rate = load_wav(jfk_wav)
    assert rate == 16000
    assert samples.dtype == np.float32
    assert samples.ndim == 1 and samples.size == 176000
    assert np.abs(samples).max() <= 1.0


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "out.wav"
    save_wav(path, np.array([0.0, 0.5, -0.5], dtype=np.float32), 22050)
    samples, rate = load_wav(path)
    assert rate == 22050
    np.testing.assert_allclose(samples, [0.0, 0.5, -0.5], atol=1e-4)


def test_extensible_24_bit_stereo_is_downmixed(tmp_path):
    # WAVE_FORMAT_EXTENSIBLE, which Python's wave module rejects before 3.12.
    frames = _int24([0x400000, 0, -0x400000, -0x400000])  # (0.5, 0), (-0.5, -0.5)
    path = _write_wav(tmp_path / "x.wav", frames, channels=2, rate=48000,
                      container_bits=24, extensible=True)
    samples, rate = load_wav(path)
    assert rate == 48000
    np.testing.assert_allclose(samples, [0.25, -0.5])


def test_extensible_24_bit_in_32_bit_container(tmp_path):
    # Valid bits are left-justified in the container.
    frames = struct.pack("<2i", 0x40000000, -0x80000000)
    path = _write_wav(tmp_path / "x.wav", frames, container_bits=32, valid_bits=24,
                      extensible=True)
    np.testing.assert_allclose(load_wav(path)[0], [0.5, -1.0])


def test_8_bit_is_unsigned(tmp_path):
    path = _write_wav(tmp_path / "x.wav", bytes([0, 128, 192]), container_bits=8)
    np.testing.assert_allclose(load_wav(path)[0], [-1.0, 0.0, 0.5])


def test_other_chunks_and_padding_are_skipped(tmp_path):
    odd_chunk = b"LIST" + struct.pack("<I", 3) + b"abc" + b"\0"  # odd size, one pad byte
    frames = struct.pack("<2h", 16384, -16384)
    path = _write_wav(tmp_path / "x.wav", frames, before_data=odd_chunk)
    np.testing.assert_allclose(load_wav(path)[0], [0.5, -0.5])


def test_streamed_size_and_truncated_frame(tmp_path):
    # A streaming writer leaves the data size at 0xFFFFFFFF; the file then ends
    # inside a frame.
    frames = struct.pack("<3h", 16384, 0, -16384)[:-1]
    path = _write_wav(tmp_path / "x.wav", frames, data_size=0xFFFFFFFF)
    np.testing.assert_allclose(load_wav(path)[0], [0.5, 0.0])


@pytest.mark.parametrize("extensible", [False, True])
def test_float_wav_is_rejected_with_a_hint(tmp_path, extensible):
    path = _write_wav(tmp_path / "x.wav", struct.pack("<f", 0.5), container_bits=32,
                      format_tag=3, extensible=extensible)
    with pytest.raises(ValueError, match="float WAV"):
        load_wav(path)


def test_not_a_wav_file(tmp_path):
    path = tmp_path / "x.wav"
    path.write_bytes(b"ID3\x04" + bytes(64))
    with pytest.raises(ValueError, match="RIFF"):
        load_wav(path)
