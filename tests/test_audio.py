import numpy as np
import pytest

from nemo_speech import load_wav
from nemo_speech._common import as_mono_f32


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
