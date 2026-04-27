"""
Unit tests for grpo_dataset. Run with:
    uv run python test_grpo_dataset.py
"""

from pathlib import Path
from PIL import Image
from datasets import Dataset

from grpo_dataset import to_grpo_format, _build_prompt_messages


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------
def _make_dummy_image(path: Path) -> None:
    """Write a tiny 4x4 RGB JPEG so tests don't depend on real VG images."""
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
    img = Image.new("RGB", (2, 2))
    msgs = _build_prompt_messages(img, "There is a duck.")
    assert isinstance(msgs, list) and len(msgs) == 1
    assert msgs[0]["role"] == "user"
    content = msgs[0]["content"]
    assert len(content) == 2
    assert content[0]["type"] == "image"
    assert content[0]["image"] is img
    assert content[1]["type"] == "text"
    assert "There is a duck." in content[1]["text"]


def test_build_prompt_uses_model_prompt_template():
    """The text portion should match SFT's MODEL_PROMPT for train/eval consistency."""
    from SFTTrain import MODEL_PROMPT

    img = Image.new("RGB", (2, 2))
    msgs = _build_prompt_messages(img, "test claim")
    text = msgs[0]["content"][1]["text"]
    assert text == MODEL_PROMPT.format(claim="test claim")


# ---------------------------------------------------------------------------
# to_grpo_format
# ---------------------------------------------------------------------------
def test_to_grpo_format_columns(tmp_path: Path):
    ds = _make_exploded_dataset(tmp_path)
    grpo_ds = to_grpo_format(ds, load_images=True)
    # Only the columns we need should remain.
    assert set(grpo_ds.column_names) == {"prompt", "label_text", "hal_type"}
    assert len(grpo_ds) == 2


def test_to_grpo_format_preserves_labels(tmp_path: Path):
    ds = _make_exploded_dataset(tmp_path)
    grpo_ds = to_grpo_format(ds, load_images=True)
    assert grpo_ds[0]["label_text"] == "CORRECT"
    assert grpo_ds[1]["label_text"] == "HALLUCINATED"
    assert grpo_ds[0]["hal_type"] == "object_existence"
    assert grpo_ds[1]["hal_type"] == "counting_error"


def test_to_grpo_format_loads_images_when_asked(tmp_path: Path):
    ds = _make_exploded_dataset(tmp_path)
    grpo_ds = to_grpo_format(ds, load_images=True)
    image_field = grpo_ds[0]["prompt"][0]["content"][0]["image"]
    # When datasets stores PIL images, they may come back as PIL.Image or as
    # bytes/dict — what we care about is that it's not the raw path string.
    assert not isinstance(image_field, str)


def test_to_grpo_format_keeps_paths_when_not_loading(tmp_path: Path):
    ds = _make_exploded_dataset(tmp_path)
    grpo_ds = to_grpo_format(ds, load_images=False)
    image_field = grpo_ds[0]["prompt"][0]["content"][0]["image"]
    assert isinstance(image_field, str)
    assert image_field.endswith(".jpg")


def test_to_grpo_format_text_contains_claim(tmp_path: Path):
    ds = _make_exploded_dataset(tmp_path)
    grpo_ds = to_grpo_format(ds, load_images=False)
    text = grpo_ds[1]["prompt"][0]["content"][1]["text"]
    assert "There are 5 dogs." in text


def test_to_grpo_format_missing_columns_raises(tmp_path: Path):
    ds = Dataset.from_list([{"img_path": "x", "claim": "y"}])  # missing label_text, hal_type
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
    import tempfile

    tests = [(k, v) for k, v in globals().items() if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                # Pass tmp_path if the test takes one, otherwise call directly.
                import inspect
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