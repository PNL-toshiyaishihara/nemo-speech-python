"""Data of the Japanese/Mandarin TTS tokenizers shipped in the package."""

import os
import pathlib

import pytest

import nemo_speech
from nemo_speech import _data

PACKAGE = pathlib.Path(nemo_speech.__file__).resolve().parent

# What upstream checks before accepting a directory.
DATA = {
    "ja": ("MAGPIE_OPENJTALK_DIC_DIR", "open_jtalk_dic", ("char.bin", "matrix.bin", "sys.dic", "unk.dic")),
    "zh": (
        "MAGPIE_MANDARIN_G2P_DIR",
        "mandarin_g2p",
        ("jieba.dict.utf8", "hmm_model.utf8", "pinyin_chars.tsv", "pinyin_phrases.tsv"),
    ),
}
LICENSES = {
    "ja": ("open_jtalk/COPYING", "mecab/COPYING", "naist-jdic/COPYING"),
    "zh": ("cppjieba/LICENSE", "limonp/LICENSE"),
}


def _built_languages():
    return [language for language, built in nemo_speech.build_info()["tts_tokenizers"].items() if built]


@pytest.mark.parametrize("language", DATA)
def test_data_ships_with_the_tokenizer(language):
    _, name, files = DATA[language]
    directory = PACKAGE / "data" / name
    if language not in _built_languages():
        assert not directory.exists()
        return
    for file in files:
        assert (directory / file).is_file(), file
    for license in LICENSES[language]:
        assert (PACKAGE / "licenses" / "third_party" / license).is_file(), license


def test_loading_the_library_points_upstream_at_the_data():
    if os.environ.get("NEMO_SPEECH_LIB_PATH"):
        pytest.skip("NEMO_SPEECH_LIB_PATH keeps the library's own data paths")
    import nemo_speech.capi.tts  # noqa: F401

    for language in _built_languages():
        variable, _, files = DATA[language]
        directory = pathlib.Path(os.environ[variable])
        assert all((directory / file).is_file() for file in files), variable


@pytest.fixture
def clean_environment(monkeypatch):
    for variable, _, _ in DATA.values():
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.delenv("NEMO_SPEECH_LIB_PATH", raising=False)
    return monkeypatch


def test_sets_unset_variables(clean_environment):
    _data.use_bundled_tokenizer_data()
    for language in _built_languages():
        variable, name, _ = DATA[language]
        assert os.environ[variable] == str(PACKAGE / "data" / name)


def test_keeps_variables_set_by_the_user(clean_environment):
    for variable, _, _ in DATA.values():
        clean_environment.setenv(variable, "user-choice")
    _data.use_bundled_tokenizer_data()
    for variable, _, _ in DATA.values():
        assert os.environ[variable] == "user-choice"


def test_library_override_keeps_its_own_paths(clean_environment, tmp_path):
    clean_environment.setenv("NEMO_SPEECH_LIB_PATH", str(tmp_path))
    _data.use_bundled_tokenizer_data()
    for variable, _, _ in DATA.values():
        assert variable not in os.environ
