"""NativeHandle: closing waits for running calls and closes dependents first."""

import gc
import threading

import pytest

from nemo_speech._common import NativeHandle


def _recorder():
    destroyed = []
    return destroyed, destroyed.append


def test_use_yields_the_handle_until_closed():
    destroyed, destroy = _recorder()
    native = NativeHandle("model", destroy, "Model")
    with native.use() as handle:
        assert handle == "model"
    native.close()
    native.close()  # idempotent
    assert destroyed == ["model"]
    with pytest.raises(RuntimeError, match="Model is closed"):
        with native.use():
            pass


def test_close_waits_for_a_running_call():
    destroyed, destroy = _recorder()
    native = NativeHandle("model", destroy, "Model")
    entered, release = threading.Event(), threading.Event()

    def call():
        with native.use():
            entered.set()
            release.wait()
            assert destroyed == []  # still alive while in use

    worker = threading.Thread(target=call)
    worker.start()
    assert entered.wait(10)
    closer = threading.Thread(target=native.close)
    closer.start()
    closer.join(0.2)
    assert closer.is_alive() and destroyed == []  # close() is waiting
    release.set()
    worker.join(10)
    closer.join(10)
    assert destroyed == ["model"]


def test_dependents_close_before_the_parent():
    order, destroy = _recorder()
    model = NativeHandle("model", destroy, "Model")
    with model.use():
        stream = NativeHandle("stream", destroy, "Stream")
        model.adopt(stream)
    model.close()
    assert order == ["stream", "model"]
    with pytest.raises(RuntimeError):
        with stream.use():
            pass


def test_closed_dependent_is_forgotten():
    order, destroy = _recorder()
    model = NativeHandle("model", destroy, "Model")
    with model.use():
        stream = NativeHandle("stream", destroy, "Stream")
        model.adopt(stream)
    stream.close()
    model.close()
    assert order == ["stream", "model"]


class _Wrapper:
    """Mimics Recognizer/RecognitionStream: finalizers that close their handle."""

    def __init__(self, native, owner=None):
        self.native = native
        self.owner = owner

    def __del__(self):
        self.native.close()


def test_cycle_collection_closes_streams_before_the_model():
    # When a model and its stream die in one reference cycle, the collector
    # finalizes them in an arbitrary order; the stream must still go first.
    for model_first in (True, False):
        order, destroy = _recorder()
        model = _Wrapper(NativeHandle("model", destroy, "Model"))
        with model.native.use():
            stream_native = NativeHandle("stream", destroy, "Stream")
            model.native.adopt(stream_native)
        stream = _Wrapper(stream_native, owner=model)
        holder = {"model": model, "stream": stream}
        holder["self"] = holder  # a cycle reaching both wrappers
        if model_first:
            model.cycle = holder
        else:
            stream.cycle = holder
        del model, stream, stream_native, holder
        gc.collect()
        assert order == ["stream", "model"], model_first
