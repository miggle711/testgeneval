"""Fail-fast check that vLLM's output-token reservation fits its context window.

Run by m3_run_inference.slurm before `vllm serve`. Stdlib only, so it can
run before anything heavy is imported.
"""

import argparse
import sys


def validate_token_limits(max_new_tokens: str, max_model_len: str) -> None:
    """Raise ValueError unless max_new_tokens is unset or below max_model_len.

    Takes the raw env var strings, an empty max_new_tokens means unset.
    Equal is rejected too: it leaves no room at all for the prompt.
    """
    if not max_new_tokens:
        return
    try:
        new_tokens, model_len = int(max_new_tokens), int(max_model_len)
    except ValueError:
        raise ValueError(
            f"MAX_NEW_TOKENS={max_new_tokens!r} and MAX_MODEL_LEN={max_model_len!r} "
            f"must both be integers."
        ) from None
    if new_tokens >= model_len:
        raise ValueError(
            f"MAX_NEW_TOKENS={new_tokens} must be less than MAX_MODEL_LEN={model_len}: "
            f"vLLM would reserve more output tokens than the whole context window "
            f"holds, which OOMs at worker startup. MAX_MODEL_LEN defaults to 32768 "
            f"when not passed, set it explicitly alongside MAX_NEW_TOKENS."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max_new_tokens", default="")
    parser.add_argument("--max_model_len", required=True)
    args = parser.parse_args()
    try:
        validate_token_limits(args.max_new_tokens, args.max_model_len)
    except ValueError as e:
        sys.exit(f"ERROR: {e}")


if __name__ == "__main__":
    main()
