"""Data files that the Japanese and Mandarin TTS tokenizers read at run time.

Upstream looks for the OpenJTalk dictionary and the Mandarin G2P tables
through an environment variable first, then in the tokenizer directory, and
last at build and install paths compiled into the library, which only exist on
the build machine. The package ships both data sets and sets the variables to
them, so the tokenizers work wherever the package is installed.
"""

from __future__ import annotations

import os
import pathlib

_DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"

# Environment variable read by upstream -> directory under nemo_speech/data.
_TOKENIZER_DATA = {
    "MAGPIE_OPENJTALK_DIC_DIR": "open_jtalk_dic",
    "MAGPIE_MANDARIN_G2P_DIR": "mandarin_g2p",
}


def use_bundled_tokenizer_data() -> None:
    """Point the TTS tokenizers at the data shipped with the package.

    Values the user has set are kept, and a library given through
    ``NEMO_SPEECH_LIB_PATH`` keeps the paths compiled into it.
    """
    if os.environ.get("NEMO_SPEECH_LIB_PATH"):
        return
    for variable, name in _TOKENIZER_DATA.items():
        directory = _DATA_DIR / name
        if not os.environ.get(variable) and directory.is_dir():
            try:
                os.environ[variable] = str(directory)
            except OSError:
                # Windows: the C runtime could not convert the path to the
                # ANSI code page. Only that language becomes unavailable.
                pass
