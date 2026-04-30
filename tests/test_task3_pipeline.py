import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import task3_pipeline as t3


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGE = ROOT / t3.LOCAL_SAMPLE_IMAGE
POPE_FILE = ROOT / "data/benchmark/pope/coco_pope_popular.json"


class Task3PipelineUnitTests(unittest.TestCase):
    def test_extract_json_object_from_fenced_output(self):
        text = 'Here is the result:\n```json\n{"claims": []}\n```'
        self.assertEqual(t3.extract_json_object(text), {"claims": []})

    def test_parse_verification_normalizes_bad_ids_and_labels(self):
        parsed = t3.parse_verification(
            json.dumps(
                {
                    "claims": [
                        {
                            "id": "not-an-int",
                            "claim": "A red car is visible.",
                            "label": "hallucination",
                            "reason": "No red car.",
                        }
                    ]
                }
            )
        )
        self.assertEqual(parsed[0].id, 1)
        self.assertEqual(parsed[0].label, "HALLUCINATED")

    def test_split_response_into_claims(self):
        claims = t3.split_response_into_claims(
            "A train is parked by the platform. The sky is blue; people wait nearby."
        )
        self.assertEqual(
            claims,
            [
                "A train is parked by the platform.",
                "The sky is blue",
                "people wait nearby.",
            ],
        )

    def test_claim_verifier_mode_calls_one_prompt_per_claim(self):
        outputs = iter(["CORRECT", "HALLUCINATED"])

        def fake_generate(*args, **kwargs):
            return next(outputs)

        with patch.object(t3, "generate_text", side_effect=fake_generate) as mocked:
            verification = t3.verify_response(
                verifier_model=object(),
                verifier_processor=object(),
                device=object(),
                image_path="image.jpg",
                response="A dog is visible. A cat is visible.",
                verifier_mode="claim",
            )

        self.assertEqual(mocked.call_count, 2)
        self.assertEqual([claim.label for claim in verification], ["CORRECT", "HALLUCINATED"])
        self.assertEqual(verification[1].claim, "A cat is visible.")

    def test_correct_response_short_circuits_when_nothing_flagged(self):
        response = "A concise visible-only caption."
        verification = [t3.VerificationClaim(id=1, claim=response, label="CORRECT")]
        with patch.object(t3, "generate_text") as mocked:
            corrected = t3.correct_response(
                generator_model=object(),
                generator_processor=object(),
                device=object(),
                image_path="image.jpg",
                response=response,
                verification=verification,
            )
        self.assertEqual(corrected, response)
        mocked.assert_not_called()

    def test_mock_loop_corrects_flagged_claims(self):
        model = t3.MockVLM()

        response, verification, corrected = t3.run_loop(
            model,
            None,
            model,
            None,
            "mock",
            str(SAMPLE_IMAGE),
            t3.DEFAULT_PROMPT,
            verifier_mode="json",
            max_new_tokens=64,
            verifier_max_new_tokens=128,
        )

        self.assertIn("red car", response)
        self.assertTrue(any(t3.is_hallucinated(claim) for claim in verification))
        self.assertEqual(corrected, "A person is skiing on snow.")

    def test_mock_best_of_n_picks_least_flagged_candidate(self):
        model = t3.MockVLM()

        response, verification = t3.run_best_of_n(
            model,
            None,
            model,
            None,
            "mock",
            str(SAMPLE_IMAGE),
            t3.DEFAULT_PROMPT,
            n=2,
            verifier_mode="json",
            max_new_tokens=64,
            verifier_max_new_tokens=128,
        )

        self.assertEqual(response, "A person is skiing on snow.")
        self.assertFalse(any(t3.is_hallucinated(claim) for claim in verification))

    def test_cli_mock_backend_writes_jsonl_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "task3_mock.jsonl"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "task3_pipeline.py"),
                    "--mock-backend",
                    "--pope-file",
                    str(POPE_FILE),
                    "--image-root",
                    str(SAMPLE_IMAGE.parent),
                    "--max-samples",
                    "1",
                    "--num-candidates",
                    "2",
                    "--skip-missing-images",
                    "--output",
                    str(output),
                ],
                check=True,
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

            records = [json.loads(line) for line in output.read_text().splitlines()]

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["corrected_response"], "A person is skiing on snow.")
        self.assertEqual(record["best_of_n_response"], "A person is skiing on snow.")
        self.assertEqual(record["meta"]["pope_question"], "Is there a snowboard in the image?")
        self.assertEqual(record["meta"]["verifier_mode"], "json")

    def test_cli_skip_missing_images_handles_partial_local_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            partial_image_root = tmp_path / "images"
            partial_image_root.mkdir()
            (partial_image_root / SAMPLE_IMAGE.name).write_bytes(SAMPLE_IMAGE.read_bytes())
            output = tmp_path / "task3_mock_partial.jsonl"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "task3_pipeline.py"),
                    "--mock-backend",
                    "--pope-file",
                    str(POPE_FILE),
                    "--image-root",
                    str(partial_image_root),
                    "--max-samples",
                    "7",
                    "--skip-missing-images",
                    "--output",
                    str(output),
                ],
                check=True,
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

            records = [json.loads(line) for line in output.read_text().splitlines()]

        self.assertEqual(len(records), 6)


if __name__ == "__main__":
    unittest.main()
