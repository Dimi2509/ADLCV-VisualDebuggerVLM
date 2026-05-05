import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import evaluate_task3_pope as ev


ROOT = Path(__file__).resolve().parents[1]


class Task3PopeEvaluatorTests(unittest.TestCase):
    def test_extract_pope_object(self):
        self.assertEqual(
            ev.extract_pope_object("Is there a dining table in the image?"),
            "dining table",
        )
        self.assertEqual(ev.extract_pope_object("Are there any skis in the image?"), "skis")

    def test_infer_pope_presence_handles_mentions_and_negation(self):
        question = "Is there a car in the image?"
        self.assertEqual(ev.infer_pope_presence("A red car is visible.", question), "yes")
        self.assertEqual(ev.infer_pope_presence("There is no visible car.", question), "no")
        self.assertEqual(ev.infer_pope_presence("A person is skiing on snow.", question), "no")

    def test_evaluate_records(self):
        records = [
            {
                "single_pass_response": "A person is skiing near a red car.",
                "verification": [
                    {"claim": "A person is skiing near a red car.", "label": "HALLUCINATED"}
                ],
                "corrected_response": "A person is skiing on snow.",
                "best_of_n_response": "A person is skiing on snow.",
                "best_of_n_verification": [
                    {"claim": "A person is skiing on snow.", "label": "CORRECT"}
                ],
                "meta": {
                    "pope_question": "Is there a car in the image?",
                    "pope_label": "no",
                    "latency_sec": 2.0,
                },
            },
            {
                "single_pass_response": "A person is skiing on snow.",
                "verification": [
                    {"claim": "A person is skiing on snow.", "label": "CORRECT"},
                    {
                        "claim": "UNPARSEABLE_VERIFICATION_OUTPUT",
                        "label": "HALLUCINATED",
                    },
                ],
                "corrected_response": "A person is skiing on snow.",
                "best_of_n_response": None,
                "best_of_n_verification": None,
                "meta": {
                    "pope_question": "Is there a person in the image?",
                    "pope_label": "yes",
                    "latency_sec": 4.0,
                },
            },
        ]

        metrics = ev.evaluate(records)

        self.assertEqual(metrics["samples"], 2)
        self.assertEqual(metrics["avg_latency_sec"], 3.0)
        self.assertEqual(metrics["correction_rate"], 0.5)
        self.assertEqual(
            metrics["verifier_flag_metrics"]["single_pass"][
                "total_flagged_hallucinated_claims"
            ],
            1,
        )
        self.assertEqual(
            metrics["verifier_flag_metrics"]["single_pass"][
                "total_unparseable_verification_outputs"
            ],
            1,
        )
        self.assertEqual(
            metrics["verifier_flag_metrics"]["single_pass"][
                "samples_with_any_flagged_hallucination"
            ],
            1,
        )
        self.assertEqual(metrics["pope_consistency_metrics"]["single_pass"]["fp"], 1)
        self.assertEqual(metrics["pope_consistency_metrics"]["corrected"]["fp"], 0)
        self.assertEqual(
            metrics["pope_consistency_metrics"]["best_of_n"]["samples_missing_response"],
            1,
        )

    def test_cli_writes_json_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "task3.jsonl"
            output_path = Path(tmp) / "metrics.json"
            input_path.write_text(
                json.dumps(
                    {
                        "single_pass_response": "A dog is visible.",
                        "verification": [{"claim": "A dog is visible.", "label": "CORRECT"}],
                        "corrected_response": "A dog is visible.",
                        "meta": {
                            "pope_question": "Is there a dog in the image?",
                            "pope_label": "yes",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "evaluate_task3_pope.py"),
                    str(input_path),
                    "--format",
                    "json",
                    "--output-json",
                    str(output_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            metrics = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(metrics["pope_consistency_metrics"]["single_pass"]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
