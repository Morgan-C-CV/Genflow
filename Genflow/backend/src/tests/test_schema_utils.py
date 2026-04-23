import unittest

from app.agent.runtime_models import NormalizedSchema
from app.agent.schema_utils import (
    MetadataParseError,
    MetadataValidationError,
    parse_and_normalize_metadata,
    parse_metadata_json,
)


VALID_METADATA_JSON = """
{
  "prompt": "a bright cinematic portrait",
  "negative_prompt": "blurry, low quality",
  "cfgscale": "7",
  "steps": "30",
  "sampler": "DPM++ 2M",
  "seed": "1234567890",
  "model": "sdxl-base",
  "clipskip": "2",
  "style": "cinematic, vivid",
  "lora": "portrait-helper, color-boost",
  "full_metadata_string": "prompt: a bright cinematic portrait"
}
"""


class SchemaUtilsTest(unittest.TestCase):
    def test_parse_metadata_json_returns_object(self):
        parsed = parse_metadata_json(VALID_METADATA_JSON)
        self.assertEqual(parsed["prompt"], "a bright cinematic portrait")

    def test_parse_and_normalize_metadata_returns_schema_object(self):
        schema = parse_and_normalize_metadata(VALID_METADATA_JSON)
        self.assertIsInstance(schema, NormalizedSchema)
        self.assertEqual(schema.model, "sdxl-base")
        self.assertEqual(schema.style, ["cinematic", "vivid"])
        self.assertEqual(schema.lora, ["portrait-helper", "color-boost"])

    def test_invalid_json_raises_parse_error(self):
        with self.assertRaises(MetadataParseError):
            parse_metadata_json("{bad json")

    def test_missing_required_field_raises_validation_error(self):
        missing_field_json = """
        {
          "prompt": "x",
          "negative_prompt": "y",
          "cfgscale": "7",
          "steps": "30",
          "sampler": "Euler",
          "seed": "1",
          "model": "m",
          "clipskip": "2",
          "style": "photo",
          "full_metadata_string": "..."
        }
        """
        with self.assertRaises(MetadataValidationError):
            parse_and_normalize_metadata(missing_field_json)

    def test_non_string_field_raises_validation_error(self):
        wrong_type_json = """
        {
          "prompt": "x",
          "negative_prompt": "y",
          "cfgscale": 7,
          "steps": "30",
          "sampler": "Euler",
          "seed": "1",
          "model": "m",
          "clipskip": "2",
          "style": "photo",
          "lora": "none",
          "full_metadata_string": "..."
        }
        """
        with self.assertRaises(MetadataValidationError):
            parse_and_normalize_metadata(wrong_type_json)


if __name__ == "__main__":
    unittest.main()
