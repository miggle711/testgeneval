# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import os
import subprocess

from swebench_docker.constants import VALID_K
from inference.prediction_paths import prediction_stem

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to run entire evaluation pipeline"
    )
    parser.add_argument(
        "--results_dir", type=str, help="Path to results directory", required=True
    )
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        required=True,
        help="HuggingFace dataset name or local path",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="API model name, HuggingFace model id, local checkpoint path, or baseline",
        required=True,
    )
    parser.add_argument(
        "--num_samples_full",
        type=int,
        help="Number of samples to run",
        choices=VALID_K,
        default=1,
    )
    parser.add_argument(
        "--num_samples_completion",
        type=int,
        help="Number of samples to run",
        choices=VALID_K,
        default=5,
    )
    parser.add_argument(
        "--namespace",
        type=str,
        help="Docker repository namespace (default: kdjain, whose images carry cosmic-ray)",
        required=False,
        default="kdjain",
    )
    parser.add_argument(
        "--num_processes", type=int, help="Number of processes to run", default=1
    )
    parser.add_argument(
        "--temperature", type=float, help="(Optional) Model temperature (default: 0.8, the rerun protocol's)", default=0.8
    )
    parser.add_argument(
        "--rerun_preds",
        action="store_true",
        help="(Optional) Rerun predictions if they already exist",
    )
    parser.add_argument(
        "--rerun_eval",
        action="store_true",
        help="(Optional) Rerun eval if they already exist",
    )
    parser.add_argument(
        "--skip_mutation", action="store_true", help="(Optional) Skip mutation"
    )
    parser.add_argument(
        "--azure", action="store_true", help="(Optional) Run with azure"
    )
    parser.add_argument(
        "--skip_full", action="store_true", help="(Optional) Skip full inference"
    )
    parser.add_argument(
        "--skip_completion",
        action="store_true",
        help="(Optional) Skip completion inference",
    )
    parser.add_argument(
        "--prompt_config",
        type=str,
        choices=["instruct", "kg_only"],
        default="instruct",
        help="Which prompt strategy to use (passed through to "
             "inference.api.run_api). Under the current test-generation "
             "scope, both 'instruct' and 'kg_only' only support 'full' "
             "-- pass --skip_completion, not --skip_full.",
    )
    parser.add_argument(
        "--kg_prompts_path",
        type=str,
        default=None,
        help="Path to pre-computed KG prompts (scripts/build_kg_prompts.py "
             "in pycodekg). Required for --prompt_config kg_only; also "
             "read by 'instruct' for its target_function/target_class "
             "focus wording.",
    )
    args = parser.parse_args()

    print(
        "NOTE: Make sure you have built the docker images for the appropriate dataset"
    )

    dataset_name_or_path = (
        os.path.abspath(args.dataset_name_or_path)
        if os.path.exists(args.dataset_name_or_path)
        else args.dataset_name_or_path
    )

    data_suf = dataset_name_or_path.split("/")[-1]
    model_suf = args.model.split("/")[-1]

    if model_suf == "Meta-Llama-3.1-405B-Instruct":
        args.model = model_suf

    API_MODELS = [
        "gpt-4o-2024-05-13",
        "gpt-4-0613",
        "gpt-4-turbo-2024-04-09",
        "gpt-3.5-turbo-0125",
        "Meta-Llama-3.1-405B-Instruct",
        "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        "llama-3.1-8b-instant",
    ]
    if args.model in API_MODELS:
        if (not args.skip_full and not args.skip_completion
                and args.num_samples_full != args.num_samples_completion):
            parser.error(
                "The API backend uses one sample count for all active settings. "
                "Use equal --num_samples_full/--num_samples_completion values, "
                "or select one setting with --skip_completion or --skip_full."
            )
        api_samples = args.num_samples_completion if args.skip_full else args.num_samples_full
    if args.model not in API_MODELS and args.model != "baseline" and args.prompt_config == "kg_only":
        parser.error("The HuggingFace backend does not support --prompt_config kg_only; "
                     "use the API runner with a served model instead.")
    output_samples = args.num_samples_completion if args.skip_full else args.num_samples_full

    print(
        f"Running pipeline for {args.model} with pass@{args.num_samples_full} (full) and pass@{args.num_samples_completion} (completion) on {data_suf}"
    )

    base_dir = os.path.join(os.path.abspath(args.results_dir), data_suf)
    print(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    log_dir = os.path.join(base_dir, "data_logs", model_suf)
    os.makedirs(log_dir, exist_ok=True)

    pred_dir = os.path.join(base_dir, "preds")
    os.makedirs(pred_dir, exist_ok=True)

    pred_output_filename = prediction_stem(
        model_suf, dataset_name_or_path, args.temperature, output_samples
    ) + ".jsonl"
    print(pred_output_filename)
    preds_file = os.path.join(pred_dir, pred_output_filename)

    if os.path.exists(preds_file) and args.rerun_preds:
        os.remove(preds_file)

    if not os.path.exists(preds_file):
        if args.model in API_MODELS:
            model_extra_cmd = ["--model_args", f"temperature={args.temperature}"]
            model_extra_cmd += ["--azure"] if args.azure else []
            model_extra_cmd += ["--skip_full"] if args.skip_full else []
            model_extra_cmd += ["--skip_completion"] if args.skip_completion else []
            model_extra_cmd += ["--prompt_config", args.prompt_config]
            # Forwarded regardless of prompt_config: run_api.py's instruct
            # path also reads this, for its target_functions/target_classes
            # focus-line wording (miggle711/pycodekg#125, this repo's #6) --
            # not just kg_only, which needs it for the prompt content itself.
            if args.kg_prompts_path is not None:
                model_extra_cmd += ["--kg_prompts_path", args.kg_prompts_path]
            # Run model prediction
            model_cmd = [
                "python",
                "-m",
                "inference.api.run_api",
                "--model_name_or_path",
                args.model,
                "--dataset_name_or_path",
                dataset_name_or_path,
                "--output_dir",
                pred_dir,
                "--num_samples",
                str(api_samples),
            ] + model_extra_cmd
            subprocess.run(model_cmd, check=True)
        elif args.model != "baseline":
            model_extra_cmd = ["--skip_full"] if args.skip_full else []
            model_extra_cmd += ["--skip_completion"] if args.skip_completion else []
            model_cmd = [
                "python",
                "-m",
                "inference.huggingface.run_huggingface",
                "--model_name_or_path",
                args.model,
                "--dataset_name_or_path",
                dataset_name_or_path,
                "--use_auth_token",
                "--output_dir",
                pred_dir,
                "--num_samples_completion",
                str(args.num_samples_completion),
                "--num_samples_generation",
                str(args.num_samples_full),
                "--temperature",
                str(args.temperature),
            ]
            model_cmd += model_extra_cmd
            subprocess.run(model_cmd, check=True)

    # Run evaluation
    extra_cmd = ["--skip_existing"] if not args.rerun_eval else []
    extra_cmd += (
        ["--skip_mutation"] if args.skip_mutation and args.model != "baseline" else []
    )
    if args.model == "baseline":
        eval_cmd = [
            "python",
            "run_evaluation_baseline.py",
            "--log_dir",
            log_dir,
            "--swe_bench_tasks",
            dataset_name_or_path,
            "--num_processes",
            str(args.num_processes),
            "--namespace",
            args.namespace,
        ] + extra_cmd
        report_cmd = [
            "python",
            "generate_report_baseline.py",
            "--log_dir",
            log_dir,
            "--output_dir",
            base_dir,
            "--swe_bench_tasks",
            dataset_name_or_path,
        ]
    else:
        eval_cmd = [
            "python",
            "run_evaluation.py",
            "--predictions_path",
            preds_file,
            "--log_dir",
            log_dir,
            "--swe_bench_tasks",
            dataset_name_or_path,
            "--num_processes",
            str(args.num_processes),
            "--namespace",
            args.namespace,
        ] + extra_cmd
        report_cmd = [
            "python",
            "generate_report.py",
            "--predictions_path",
            preds_file,
            "--log_dir",
            log_dir,
            "--output_dir",
            base_dir,
            "--swe_bench_tasks",
            dataset_name_or_path,
        ]

    subprocess.run(eval_cmd, check=True)
    subprocess.run(report_cmd, check=True)
