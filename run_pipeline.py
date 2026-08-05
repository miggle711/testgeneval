# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import os
import subprocess

from swebench_docker.constants import VALID_K

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
        help="Model name",
        choices=[
            "gpt-4o-2024-05-13",
            "gpt-4-0613",
            "gpt-4-turbo-2024-04-09",
            "gpt-3.5-turbo-0125",
            "meta-llama/CodeLlama-7b-Instruct-hf",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "meta-llama/CodeLlama-70b-Instruct-hf",
            "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "meta-llama/Meta-Llama-3.1-405B-Instruct",
            "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
            "mistralai/Codestral-22B-v0.1",
            "google/gemma-2-9b-it",
            "google/gemma-2-27b-it",
            "baseline",
            "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        ],
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
        help="Docker repository namespace",
        required=False,
        default="aorwall",
    )
    parser.add_argument(
        "--num_processes", type=int, help="Number of processes to run", default=1
    )
    parser.add_argument(
        "--temperature", type=float, help="(Optional) Model temperature", default=0.2
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

    pred_output_filename = f"{model_suf}__{data_suf}__{args.temperature}__test.jsonl"
    print(pred_output_filename)
    preds_file = os.path.join(pred_dir, pred_output_filename)

    if os.path.exists(preds_file) and args.rerun_preds:
        os.remove(preds_file)

    API_MODELS = [
        "gpt-4o-2024-05-13",
        "gpt-4-0613",
        "gpt-4-turbo-2024-04-09",
        "gpt-3.5-turbo-0125",
        "Meta-Llama-3.1-405B-Instruct",
        "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    ]
    if not os.path.exists(preds_file):
        if args.model in API_MODELS:
            model_extra_cmd = ["--model_args", f"temperature={args.temperature}"]
            model_extra_cmd += ["--azure"] if args.azure else []
            model_extra_cmd += ["--skip_full"] if args.skip_full else []
            model_extra_cmd += ["--skip_completion"] if args.skip_completion else []
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
                "--num_samples_full",
                str(args.num_samples_full),
                "--num_samples_completion",
                str(args.num_samples_completion),
            ] + model_extra_cmd
            subprocess.run(model_cmd)
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
                "--num_samples_full",
                str(args.num_samples_full),
                "--temperature",
                str(args.temperature),
            ]
            model_cmd += model_extra_cmd
            subprocess.run(model_cmd)

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

    subprocess.run(eval_cmd)
    subprocess.run(report_cmd)
