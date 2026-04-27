"""
Unit tests for grpo_rewards. Run with:
    uv run python -m pytest test_grpo_rewards.py -v
or just:
    uv run python test_grpo_rewards.py
"""

from grpo_rewards import compute_reward, make_reward_func


# ---------------------------------------------------------------------------
# compute_reward
# ---------------------------------------------------------------------------
def test_weighted_true_positive():
    assert compute_reward("HALLUCINATED", "HALLUCINATED") == 1.5


def test_weighted_true_negative():
    assert compute_reward("CORRECT", "CORRECT") == 1.0


def test_weighted_false_positive():
    assert compute_reward("HALLUCINATED", "CORRECT") == -1.0


def test_weighted_false_negative():
    assert compute_reward("CORRECT", "HALLUCINATED") == -1.5


def test_format_violation_unknown():
    assert compute_reward("UNKNOWN", "HALLUCINATED") == -2.0


def test_format_violation_garbage():
    assert compute_reward("I think it might be wrong", "CORRECT") == -2.0


def test_unweighted_correct():
    assert compute_reward("HALLUCINATED", "HALLUCINATED", weighted=False) == 1.0
    assert compute_reward("CORRECT", "CORRECT", weighted=False) == 1.0


def test_unweighted_incorrect():
    assert compute_reward("HALLUCINATED", "CORRECT", weighted=False) == -1.0
    assert compute_reward("CORRECT", "HALLUCINATED", weighted=False) == -1.0


def test_unweighted_format_still_punished():
    # Even in unweighted mode we keep -2.0 for format violations.
    assert compute_reward("UNKNOWN", "CORRECT", weighted=False) == -2.0


# ---------------------------------------------------------------------------
# make_reward_func — TRL-style signature with conversational completions
# ---------------------------------------------------------------------------
def test_reward_func_conversational():
    reward_fn = make_reward_func(weighted=True)
    completions = [
        [{"role": "assistant", "content": "HALLUCINATED"}],
        [{"role": "assistant", "content": "CORRECT"}],
        [{"role": "assistant", "content": "HALLUCINATED"}],
        [{"role": "assistant", "content": "definitely not sure"}],
    ]
    label_text = ["HALLUCINATED", "CORRECT", "CORRECT", "HALLUCINATED"]
    rewards = reward_fn(completions, label_text=label_text)
    assert rewards == [1.5, 1.0, -1.0, -2.0]


def test_reward_func_string_completions():
    """Standard (non-conversational) format must also work."""
    reward_fn = make_reward_func(weighted=False)
    completions = ["CORRECT", "HALLUCINATED", "HALLUCINATED"]
    label_text = ["CORRECT", "CORRECT", "HALLUCINATED"]
    rewards = reward_fn(completions, label_text=label_text)
    assert rewards == [1.0, -1.0, 1.0]


def test_reward_func_lowercase_labels_normalised():
    """parse_predicted_label upper-cases internally, so lowercase output works."""
    reward_fn = make_reward_func(weighted=True)
    completions = [[{"role": "assistant", "content": "hallucinated"}]]
    rewards = reward_fn(completions, label_text=["HALLUCINATED"])
    assert rewards == [1.5]


def test_reward_func_missing_label_column_raises():
    reward_fn = make_reward_func()
    try:
        reward_fn(["CORRECT"], some_other_column=["x"])
    except KeyError as e:
        assert "label_text" in str(e)
    else:
        raise AssertionError("Expected KeyError when label_text is missing")


def test_reward_func_length_mismatch_raises():
    reward_fn = make_reward_func()
    try:
        reward_fn(["CORRECT", "CORRECT"], label_text=["CORRECT"])
    except ValueError as e:
        assert "length mismatch" in str(e)
    else:
        raise AssertionError("Expected ValueError on length mismatch")


# ---------------------------------------------------------------------------
# Manual run entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            print(f"FAIL  {t.__name__}  {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {t.__name__}  {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(0 if failed == 0 else 1)