import json
import tempfile
import unittest
from pathlib import Path

import src.data_checks as checks


ROOT = Path(__file__).resolve().parents[1]


class FishSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = checks.load_data(ROOT / 'assets' / 'data' / 'Fish.csv')

    def test_source_dataset_matches_schema(self):
        self.assertEqual(checks.validate_fish_schema(self.data), [])

    def test_weight_outside_declared_range_is_rejected(self):
        changed = self.data.copy()
        changed.loc[0, 'Weight'] = 2001
        errors = checks.validate_fish_schema(changed)
        self.assertTrue(any('Weight is above maximum' in error for error in errors))

    def test_weight_rescaled_to_kilograms_is_rejected(self):
        changed = self.data.copy()
        changed['Weight'] = changed['Weight'] / 1000
        errors = checks.validate_fish_schema(changed)
        self.assertTrue(any('Weight positive values are below' in error for error in errors))

    def test_schema_unit_change_is_rejected(self):
        schema = json.loads(checks.SCHEMA_PATH.read_text(encoding='utf-8'))
        schema['properties']['Weight']['x-unit'] = 'kg'
        with tempfile.TemporaryDirectory() as directory:
            temporary_schema = Path(directory) / 'fish_schema.json'
            temporary_schema.write_text(json.dumps(schema), encoding='utf-8')
            errors = checks.validate_fish_schema(self.data, temporary_schema)
        self.assertIn("Weight unit must be 'g'", errors)


if __name__ == '__main__':
    unittest.main()