"""Low-level ctypes bindings for the NeMo-Speech.cpp C ABI.

Each submodule mirrors one header under ``include/nemo_speech/`` one-to-one:
struct, enum and function names match the C declarations, so the C
documentation applies unchanged. Prefer the high-level API in
:mod:`nemo_speech` unless you need direct control over handles.

Paths are UTF-8 (``os.fsencode`` on Windows). Wrap the ``*_create`` calls in
:func:`utf8_file_paths` so that components opening files with narrow C
runtime APIs also accept non-ASCII paths on Windows.
"""

from __future__ import annotations

import ctypes
import functools
from typing import Any, Callable, Sequence

from .._paths import utf8_file_paths

__all__ = ["SizedStructure", "ctypes_function_for", "utf8_file_paths"]


class SizedStructure(ctypes.Structure):
    """Base for the ABI's size-prefixed structs.

    ``size`` defaults to ``sizeof`` of the Python definition, as the C ABI
    requires; every other field is zero-initialized.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not self.size:
            self.size = ctypes.sizeof(self)


def ctypes_function_for(lib: ctypes.CDLL):
    """Return a decorator binding a Python stub to ``lib``'s C function.

    The decorated stub only documents the signature; calls go straight to the
    C function, which has ``argtypes``/``restype`` set from the decorator.
    """

    def ctypes_function(
        name: str, argtypes: Sequence[Any], restype: Any
    ) -> Callable[[Callable[..., Any]], Any]:
        def decorator(stub: Callable[..., Any]) -> Any:
            func = getattr(lib, name)
            func.argtypes = list(argtypes)
            func.restype = restype
            functools.wraps(stub)(func)
            return func

        return decorator

    return ctypes_function
