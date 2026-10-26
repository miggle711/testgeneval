# TestGenEval: A Large Scale Test Generation Benchmark

This is a fork of [facebookresearch/testgeneval](https://github.com/facebookresearch/testgeneval), extended as part of a research project on patch anchored knowledge graph retrieval for LLM based test generation. Three main additions over upstream.

- A `kg_only` prompt arm that replaces the whole code file with structural context retrieved from a repository knowledge graph, built with our separate [pycodekg](https://github.com/miggle711/pycodekg) tool. See "Knowledge graph guided generation" below.
- A local vLLM inference path. Upstream's inference client assumes the OpenAI API. `run_api.py` also accepts `LOCAL_MODEL_BASE_URL` to talk to any OpenAI compatible endpoint, including a self hosted vLLM server, with its own real per model context length and output token limit registry (several entries fix real truncation we hit in practice, see the comments in `inference/api/run_api.py` for specifics).
- An Apptainer based inference and evaluation backend, for running on HPC clusters (we use Monash's M3) where Docker is not available and there is no root access. `m3_run_inference.slurm` starts a real `vllm serve` process on the allocated GPU node, waits for it to become ready, then runs inference against it in the same job. Evaluation runs the same way against pre converted `.sif` images instead of Docker containers. See "Running on HPC with Apptainer" below. The original Docker based flow still works unchanged if you are not on an HPC cluster.

See our paper for the full methodology and results.

TestGenEval consists of 1,210 code test file pairs from 11 large, well-maintained repositories (3,523-78,287 stars). We use these file pairs to construct two testing tasks: 1) unit test completion for the first, last and additional tests and 2) full file unit test generation. Our benchmark is easy to run and extend, as we have docker containers for each version of each repository with coverage and mutation testing dependencies installed. For both task we use execution based metrics, including pass@1, pass@5 along with code coverage improvement, and mutation score improvement compared to the gold (human written) tests. Code and test files in \benchmark are long in length (on average 782 LOC per code file and 677 LOC per test file) and high coverage (median coverage of 60.4\%).

We measure the following metrics for the test completion task:
- pass@k (k = 1, 5)
- coverage improvement (how much generated test improves existing coverage)
- coverage improvement@pass (coverage improvement averaged only over passing tests)
- average pass@5

We measure the following metrics for the test generation task:
- pass@1
- all pass@1 (all tests generated in suite pass)
- coverage (coverage of generated tests)
- coverage@pass (coverage of generated tests for passing examples)
- mutation score (mutation score of generated tests)
- mutation score@pass (mutation score of generated tests for passing examples)

## Datasets

### TestGenEvalLite
Docker images for testbeds used in the `TestGenEvalLite` dataset has been built and tested.

### TestGenEval
Docker images for testbeds used in the `TestGenEval` dataset has been built and tested.

## Setup

To setup the repository run
```
git clone git@github.com:miggle711/testgeneval.git
cd testgeneval
conda env create -f testgeneval.yaml
conda activate testgeneval
```

Modify the `.env_template` file with the appropriate values and rename it to `.env` (specifically make sure to set SWEBENCH_DOCKER_FORK_DIR to the current directory where the repository was cloned)

**The env template setup is important, make sure you do this**

## Building TestGenEval

To build the docker images locally (adapted from [SWEBench Docker](https://github.com/aorwall/SWE-bench-docker/tree/main/docker)) run one of these commands:

**TestGenEvalLite** - TestGenEvalLite for faster evaluation
```
make -f Makefile.testgenevallite
```

**TestGenEval** - full TestGenEval (takes hours to a full day to build)
```
make -f Makefile.testgeneval
```

**OR** 

You can simply just run with images pushed to Dockerhub

To pull all images (TestGenEval) run
```
python scripts/pull_images.py --makefile Makefile.testgeneval
```

To pull lite images (TestGenEvalLite) run
```
python scripts/pull_images.py --makefile Makefile.testgenevallite
```

## Running on HPC with Apptainer

If you are on an HPC cluster without Docker or root access (we use Monash's M3), use the Apptainer based backend instead of the Docker flow above.

First convert the testbed Docker images you need to `.sif` files, which does not need root:

```
python scripts/pull_apptainer_images.py --makefile Makefile.testgenevallite
```

Then run inference and evaluation through the SLURM scripts, which set the environment each stage needs and run vLLM and Apptainer directly:

```
sbatch m3_run_inference.slurm
sbatch m3_run_evaluation.slurm
```

`m3_run_inference.slurm` launches `vllm serve` on the allocated GPU node, waits for the server to become ready, then runs `run_api.py` against it over `LOCAL_MODEL_BASE_URL`, all within the one job. `m3_run_evaluation.slurm` runs the same evaluation flow as the Docker path, against the `.sif` images instead of Docker containers. See the comments at the top of `swebench_docker/run_apptainer.py` for real, confirmed gotchas on this path (rootless pulls, cache directory placement, entrypoint resolution differences between base images).

## TestGenEval Datasets

The TestGenEval datasets are available on huggingface:
- [kjain14/testgeneval](https://huggingface.co/datasets/kjain14/testgeneval)
- [kjain14/testgenevallite](https://huggingface.co/datasets/kjain14/testgenevallite)

## Running TestGenEval

Running TestGenEval is relatively simple.

There is a python script that will run both prediction and inference.

If you built docker images locally:

```
python run_pipeline.py \
--results_dir results \
--dataset_name_or_path kjain14/testgenevallite \
--model meta-llama/Meta-Llama-3.1-8B-Instruct
```

Otherwise to pull from Dockerhub:

```
python run_pipeline.py \
--results_dir results \
--dataset_name_or_path kjain14/testgenevallite \
--model meta-llama/Meta-Llama-3.1-8B-Instruct \
--namespace kdjain
```

## Knowledge graph guided generation

To use the `kg_only` arm, pass `--prompt_config kg_only` to `run_pipeline.py` along with the path to a prebuilt KG prompts file:

```
python run_pipeline.py \
--results_dir results \
--dataset_name_or_path kjain14/testgenevallite \
--model meta-llama/Meta-Llama-3.1-8B-Instruct \
--prompt_config kg_only \
--kg_prompts_path kg_prompts.json
```

`kg_prompts.json` is built separately with [pycodekg](https://github.com/miggle711/pycodekg)'s `build_kg_prompts.py`, which constructs a repository knowledge graph for the target commit and serialises, per instance, the retrieved context that replaces the whole code file. `--kg_prompts_path` is also read by the default `instruct` config, to source the target function or target class focus wording for its prompt. Under the current scope, both `instruct` and `kg_only` only support the `full` setting, so run with `--skip_completion` rather than `--skip_full`.

To use the local vLLM inference path instead of a hosted OpenAI-compatible endpoint, set `LOCAL_MODEL_BASE_URL` (for example `http://127.0.0.1:8000/v1/`) before calling `run_api.py` directly, or let `m3_run_inference.slurm` start `vllm serve` and set it for you.

## Adding a new model to TestGenEval

Adding a new model is quite simple. Under `inference/configs` create a new file with the system prompts and the function to add prompts to the dataset.

`add_prompts_to_dataset` should output a prompt for all four settings: `full`, `first`, `last`, `extra`. The `preds_context` attribute of each datapoint contains the preamble of the file, the first test, the file without the last test and the file with the last test (full file)

Once you update this file our standard evaluation flow will work.

## TestGenEval creation

All creation scripts are housed in the `creation` subdirectory.

`transform_swebench.py` is the main script that takes the SWEBench dataset and converts it for test generation.

`filter_unittests.py` takes the baseline results and filters out datapoints with no coverage of gold tests (gold tests must cover the code under test).

## Licensing

The majority of code in this repository is licensed under CC-by-NC, however the third party code/files may be subject to different licenses.
