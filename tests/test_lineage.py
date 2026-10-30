"""Alignment between the stages (src/METHOD/_common/lineage.py)."""
import pytest

from METHOD._common.lineage import check_r_uv_identity, model_name, pick, resolve_r_uv_path


def test_pick_takes_the_upstream_value_and_rejects_a_conflicting_config_value():
    assert pick("coeff", None, 2.0, "sel.json") == 2.0  # not set -> upstream
    assert pick("coeff", 2, 2.0, "sel.json") == 2.0  # the same value (int vs float) is fine
    assert pick("layers", [1, 2], (1, 2), "sel.json") == (1, 2)
    with pytest.raises(ValueError, match="disagrees"):
        pick("coeff", 1.0, 2.0, "sel.json")
    assert pick("coeff", 3.0, None, "the config") == 3.0  # no upstream file -> the config value
    with pytest.raises(ValueError, match="neither"):
        pick("coeff", None, None, "the config")


def test_identity_check_rejects_another_forget_split_size_or_model_and_old_files():
    saved = {"forget_split": "forget10", "num_forget": 400, "model": "m/full"}
    check_r_uv_identity(saved, forget_split="forget10", num_forget=400, model="m/full")
    check_r_uv_identity(saved, forget_split=None, num_forget=None, model=None)  # nothing to compare
    with pytest.raises(ValueError, match="forget_split"):
        check_r_uv_identity(saved, forget_split="forget05")
    with pytest.raises(ValueError, match="num_forget"):
        check_r_uv_identity(saved, num_forget=200)
    with pytest.raises(ValueError, match="model"):
        check_r_uv_identity(saved, model="m/retain")
    with pytest.raises(ValueError, match="run r_uv.py again"):
        check_r_uv_identity({"num_forget": 400}, forget_split="forget10")


def test_resolve_r_uv_path_falls_back_to_the_file_next_to_the_setting_folder(tmp_path):
    (tmp_path / "r_uv.pt").write_bytes(b"")
    selection = tmp_path / "k5_c1_eoi" / "layer_selection.json"
    selection.parent.mkdir()
    assert resolve_r_uv_path(str(tmp_path / "r_uv.pt"), selection) == tmp_path / "r_uv.pt"
    assert resolve_r_uv_path("/moved/away/r_uv.pt", selection) == tmp_path / "r_uv.pt"
    with pytest.raises(FileNotFoundError):
        resolve_r_uv_path("/moved/away/r_uv.pt", tmp_path / "x" / "y" / "layer_selection.json")


def test_model_name():
    assert model_name({"pretrained": {"name_or_path": "a/b"}}) == "a/b"
