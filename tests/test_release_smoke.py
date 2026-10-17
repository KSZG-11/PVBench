import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluators.dimension_categories import (  # noqa: E402
    DIMENSION_CATEGORIES_META,
    DIMENSION_TO_CATEGORY,
)


class ReleaseSmokeTests(unittest.TestCase):
    def test_taxonomy_has_46_unique_dimensions_in_seven_domains(self):
        self.assertEqual(len(DIMENSION_TO_CATEGORY), 46)
        self.assertEqual(set(DIMENSION_CATEGORIES_META), {f"D{i}" for i in range(1, 8)})
        self.assertTrue(set(DIMENSION_TO_CATEGORY.values()).issubset(DIMENSION_CATEGORIES_META))

    def test_example_manifest_is_valid_jsonl_and_portable_schema(self):
        path = ROOT / "examples" / "manifest.example.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        for key in ("prompt_id", "prompt", "video", "modality"):
            self.assertIn(key, row)
        self.assertIn(row["modality"], {"T2V", "I2V", "CI2V"})
        self.assertTrue(row["video"].startswith("/path/to/"))
        self.assertTrue(set(row["dimensions"]).issubset(DIMENSION_TO_CATEGORY))

    def test_mit_license_and_release_note_are_present(self):
        self.assertTrue((ROOT / "LICENSE").exists())
        self.assertIn("MIT License", (ROOT / "LICENSE").read_text(encoding="utf-8-sig"))
        readme = (ROOT / "README.md").read_text(encoding="utf-8-sig")
        self.assertIn("patent", readme.lower())
        self.assertIn("Qwen3.8-27B", readme)


if __name__ == "__main__":
    unittest.main()
