import unittest

from inference.check_token_limits import validate_token_limits


class TestValidateTokenLimits(unittest.TestCase):
    def test_output_reservation_must_fit_inside_context_window(self):
        rejected = [
            ("48000", "32768"),  # the real Llama-4-Scout OOM: slurm default MAX_MODEL_LEN
            ("32768", "32768"),  # leaves no room for the prompt
            ("48k", "65536"),
        ]
        for max_new_tokens, max_model_len in rejected:
            with self.subTest(max_new_tokens=max_new_tokens, max_model_len=max_model_len):
                with self.assertRaises(ValueError):
                    validate_token_limits(max_new_tokens, max_model_len)

    def test_unset_or_smaller_max_new_tokens_passes(self):
        for max_new_tokens, max_model_len in [("", "32768"), ("48000", "65536")]:
            with self.subTest(max_new_tokens=max_new_tokens, max_model_len=max_model_len):
                validate_token_limits(max_new_tokens, max_model_len)


if __name__ == "__main__":
    unittest.main()
