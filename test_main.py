import os
import unittest
from unittest.mock import Mock, patch

import main


def response(content, model="test/model"):
    result = Mock(status_code=200, headers={})
    result.raise_for_status.return_value = None
    result.json.return_value = {
        "model": model,
        "choices": [{"message": {"content": content}}],
    }
    return result


class ModelFallbackTests(unittest.TestCase):
    def test_frontend_response_has_no_duplicate_data_wrapper(self):
        response = main.build_frontend_response(
            {
                "Prescription": "1000020419",
                "Name": "Test Patient",
                "Address": None,
                "Mobile": None,
                "Gender": None,
                "Age": 30,
                "Medicines": [{"Name": "TEST MEDICINE", "Qty": 8}],
            },
            "request",
        )

        self.assertNotIn("data", response)
        self.assertEqual(response["prescriptionNumber"], "1000020419")
        self.assertEqual(response["patient"]["name"], "Test Patient")
        self.assertEqual(response["medicines"][0]["quantity"], 8)

    def test_prompt_excludes_attendant_phone_numbers(self):
        prompt = main.EXTRACTION_PROMPT
        self.assertIn("patient's phone/mobile number only", prompt)
        self.assertIn("Beneficiary/Attendant Signature", prompt)
        self.assertIn("return Mobile as null", prompt)
        self.assertNotIn("even if it appears near signature or attendant details", prompt)

    def test_direct_google_is_called_before_nvidia(self):
        valid = '{"Name":"Google Result","Medicines":[]}'
        with (
            patch.dict(
                os.environ,
                {"NVIDIA_API_KEY": "n", "GEMINI_API_KEY": "g", "OPENROUTER_API_KEY": ""},
            ),
            patch.object(main, "get_nvidia_models", return_value=["nvidia/test"]),
            patch.object(main, "GOOGLE_MODELS", ["gemini-3.6-flash"]),
            patch.object(main.requests, "post", return_value=response(valid)) as post,
        ):
            result = main.call_vision_models(b"image", None, "request", 1)

        self.assertEqual(result["Name"], "Google Result")
        self.assertEqual(post.call_count, 1)
        self.assertIn("generativelanguage.googleapis.com", post.call_args.args[0])

    def test_null_output_falls_back_to_openrouter(self):
        valid = '{"Name":"Test Patient","Medicines":[]}'
        with (
            patch.dict(
                os.environ,
                {"NVIDIA_API_KEY": "n", "GEMINI_API_KEY": "", "OPENROUTER_API_KEY": "o"},
            ),
            patch.object(main, "get_nvidia_models", return_value=["nvidia/test"]),
            patch.object(main, "OPENROUTER_MODELS", ["google/gemini-3.8-flash"]),
            patch.object(main.requests, "post", side_effect=[response(None), response(None), response(valid)]) as post,
        ):
            result = main.call_vision_models(b"image", None, "request", 1)

        self.assertEqual(result["Name"], "Test Patient")
        self.assertEqual(post.call_count, 3)
        self.assertIn("openrouter.ai", post.call_args.args[0])

    def test_provider_specific_reasoning_settings(self):
        nvidia = main.build_model_payload("nvidia/nemotron-3.5-test", "abc", None, True, "nvidia")
        google = main.build_model_payload("gemini-3.6-flash", "abc", None, True, "google")
        openrouter = main.build_model_payload("google/gemini-3.8-flash", "abc", None, True, "openrouter")

        self.assertEqual(nvidia["chat_template_kwargs"], {"enable_thinking": False})
        self.assertNotIn("temperature", google)
        self.assertNotIn("top_p", google)
        self.assertEqual(google["max_tokens"], 4096)
        self.assertEqual(openrouter["reasoning"], {"effort": "low"})
        self.assertTrue(openrouter["provider"]["allow_fallbacks"])


if __name__ == "__main__":
    unittest.main()
