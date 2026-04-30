"""
Unit tests for grpo_dataset. Run with:
    uv run python test_grpo_dataset.py
"""

from pathlib import Path
from PIL import Image
from datasets import Dataset

from grpo_dataset import to_grpo_format, _build_prompt_messages


def _make_dummy_image(path: Path) -> None:
    Image.new("RGB", (4, 4), color=(123, 222, 64)).save(path, format="JPEG")


def _make_exploded_dataset(tmp_dir: Path) -> Dataset:
    img_a = tmp_dir / "1.jpg"
    img_b = tmp_dir / "2.jpg"
    _make_dummy_image(img_a)
    _make_dummy_image(img_b)
    rows = [
        {
            "image_id": 1,
            "img_path": str(img_a),
            "claim": "There is a cat.",
            "label_text": "CORRECT",
            "hal_type": "object_existence",
        },
        {
            "image_id": 2,
            "img_path": str(img_b),
            "claim": "There are 5 dogs.",
            "label_text": "HALLUCINATED",
            "hal_type": "counting_error",
        },
    ]
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# _build_prompt_messages
# ---------------------------------------------------------------------------
def test_build_prompt_messages_structure():
    """prompt should have one user message with an image placeholder + text."""
    msgs = _build_prompt_messages("There is a duck.")
    assert isinstance(msgs, list) and len(msgs) == 1
    assert msgs[0]["role"] == "user"
    content = msgs[0]["content"]
    assert len(content) == 2
    assert content[0]["type"] == "image"
    # CRITICAL: no "image" key at this stage — image is bound separately.
    assert "image" not in content[0]
    assert content[1]["type"] == "text"
    assert "There is a duck." in content[1]["text"]


def test_build_prompt_uses_model_prompt_template():
    from SFTTrain import MODEL_PROMPT

    msgs = _build_prompt_messages("test claim")
    text = msgs[0]["content"][1]["text"]
    assert text == MODEL_PROMPT.format(claim="test claim")


# ---------------------------------------------------------------------------
# to_grpo_format
# ---------------------------------------------------------------------------
def test_to_grpo_format_columns(tmp_path: Path):
    ds = _make_exploded_dataset(tmp_path)
    grpo_ds = to_grpo_format(ds)
    assert set(grpo_ds.column_names) == {"prompt", "image", "label_text", "hal_type"}
    assert len(grpo_ds) == 2


def test_to_grpo_format_preserves_labels(tmp_path: Path):
    ds = _make_exploded_dataset(tmp_path)
    grpo_ds = to_grpo_format(ds)
    assert grpo_ds[0]["label_text"] == "CORRECT"
    assert grpo_ds[1]["label_text"] == "HALLUCINATED"
    assert grpo_ds[0]["hal_type"] == "object_existence"
    assert grpo_ds[1]["hal_type"] == "counting_error"


def test_to_grpo_format_image_is_pil(tmp_path: Path):
    """datasets.Image() feature should decode paths back into PIL images on access."""
    ds = _make_exploded_dataset(tmp_path)
    grpo_ds = to_grpo_format(ds)
    img = grpo_ds[0]["image"]
    assert isinstance(img, Image.Image)
    assert img.size == (4, 4)


def test_to_grpo_format_text_contains_claim(tmp_path: Path):
    ds = _make_exploded_dataset(tmp_path)
    grpo_ds = to_grpo_format(ds)
    text = grpo_ds[1]["prompt"][0]["content"][1]["text"]
    assert "There are 5 dogs." in text


def test_to_grpo_format_missing_columns_raises(tmp_path: Path):
    ds = Dataset.from_list([{"img_path": "x", "claim": "y"}])
    try:
        to_grpo_format(ds)
    except ValueError as e:
        assert "missing" in str(e).lower()
    else:
        raise AssertionError("Expected ValueError on missing columns")


# ---------------------------------------------------------------------------
# Manual run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import inspect
    import tempfile

    tests = [(k, v) for k, v in globals().items() if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                params = inspect.signature(fn).parameters
                if "tmp_path" in params:
                    fn(tmp)
                else:
                    fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            print(f"FAIL  {name}  {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {name}  {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(0 if failed == 0 else 1)