# Development Log

Real problems, root causes and fixes, written as they occur.

## 2026-10-18 - Dataset license is CC BY-NC 4.0, not MIT

- What happened: the initial project plan stated that the English-Nuer-Dinka parallel corpus is under the MIT license and was published in 2025.
- Root cause: the plan was written before the dataset card was checked.
- Fix: the Hugging Face repository `Tajnaam/english-nuer-dinka-parallel-corpus` redirects to `dayomtechnologies/english-nuer-dinka-parallel-corpus`. Its card states CC BY-NC 4.0 (no commercial use, attribution required) and the repository was created on 2026-07-01. The plan was corrected. The base model NLLB-200 is also CC BY-NC 4.0, so the project stays non-commercial: no dataset or weights in the repository, no public service, fine-tuned model not published.
- Open point: the dataset is gated (conditions must be accepted with a Hugging Face account), so the pair count, the fields and the splits are verified in the data stage, not before.

## 2026-10-18 - Pre-push hook content checks did not block

- What happened: the first test run of the pre-push hook let through an access key, an ARN with an account ID, a foreign e-mail address, an attribution trailer, a Hugging Face token and an endpoint URL. Only the path and author checks blocked.
- Root cause: the scan function was called at the end of a pipeline (`git show ... | scan_text`), so it ran in a subshell and its `fail=1` was lost when the subshell exited.
- Fix: feed the function with process substitution (`scan_text label < <(git show ...)`) so it runs in the hook's own shell. All 15 test cases then behaved as expected (`proof/01-pre-push-hook-test.txt`).

## 2026-10-18 - No venv module for Python 3.14

- What happened: `python3 -m venv .venv` failed: ensurepip is not available.
- Root cause: Ubuntu ships the venv module in a separate package, python3.14-venv, which was not installed.
- Fix: the package was installed with sudo, then the project virtual environment was created. Tools that need older Python wheels (fastText for GlotLID) run in a python:3.11-slim container instead.

## 2026-10-18 - Hugging Face token rejected, then dataset file refused

- What happened: the stored token was reported invalid; after it was fixed, the dataset card downloaded but the data file returned 403 "not in the authorized list".
- Root cause: the token had been pasted four times into the token file (148 characters instead of 37), found by checking the length and the repetition without printing the token. The 403 came from the dataset access not yet being granted to the account.
- Fix: the token file was rewritten with one paste (length checked), and access to the gated dataset was granted on the dataset page. The download then succeeded.

## 2026-10-18 - The corpus is very probably machine translated

- What happened: inspection showed looping output in 7.3% of the Dinka rows (1.3% Nuer, 0.1% English), encyclopedic English source sentences, and full three-way coverage of 499,880 rows. The card does not state how the translations were made; the same publisher's health dataset states that it used Google Translate.
- Consequence: fine-tuning on this corpus teaches the model to imitate the system that produced it, and BLEU and chrF on a held-out split measure agreement with that system, not with human translation. This is stated with every metric.

## 2026-10-18 - .gitignore hid the data stack

- What happened: the commit of the data stage left out `infra/data/` (Terraform and Athena queries).
- Root cause: the `.gitignore` rule `data/` (meant for the downloaded dataset) matches a directory named `data` at any depth, including `infra/data/`.
- Fix: the dataset rules are anchored to the repository root (`/data/`, `/models/`, ...). `git check-ignore` confirms that `data/raw/` stays ignored and `infra/data/*.tf` is tracked.

## 2026-10-18 - No space left on device while installing the SageMaker SDK

- What happened: `pip install` failed with "No space left on device" on a disk with 928 GB free.
- Root cause: `/tmp` is a 3.9 GB tmpfs (memory) in this WSL setup; a 1.6 GB language-identification model left in a temporary folder filled it.
- Fix: the model was deleted after use; large downloads now go to a directory on disk.

## 2026-10-18 - SageMaker SDK install did not resolve

- What happened: `pip install sagemaker` ran for more than 10 minutes on Python 3.14 and again in a Python 3.12 container (2.4 GB downloaded) without finishing.
- Root cause: the unpinned install resolves the 3.x line and backtracks through many dependency versions.
- Fix: a tooling image (`pipelines/Dockerfile`) pins `sagemaker==2.257.7`, the last 2.x release; it installs in about a minute. The 2.x line prints a deprecation warning; the pin is recorded in docs/DECISIONS.md.

## 2026-10-18 - Commands passed to the training image were mangled

- What happened: `docker run <training image> sh -c "pip install ...; pytest"` printed the pip usage text and then "No module named pytest".
- Root cause: the image entrypoint re-wraps the command, so the `sh -c` string was not passed as one argument.
- Fix: `--entrypoint bash` with `-c`. The 8 training and evaluation tests then passed in the image (torch 2.8.0, transformers 4.56.2).

## 2026-10-19 - Processing job rejected: entrypoint longer than 256 characters

- What happened: `create_processing_job` returned a ValidationException; the `bash -c "<pip install && python run_eval.py ...>"` entrypoint string exceeded the 256-character limit of `ContainerEntrypoint` members. No job was created.
- Root cause: the whole command line, including the pinned model revision, was passed as one entrypoint member.
- Fix: a small `evaluation/run_eval.sh` is shipped with the code and used as the entrypoint; the model arguments go in `ContainerArguments`. The job then started on the first try.

## 2026-10-19 - Docker Desktop stopped between sessions

- What happened: `docker` was not found in WSL ("could not be found in this WSL 2 distro") when generating the reference pipeline definition.
- Root cause: Docker Desktop was not running, so its CLI link in WSL was missing.
- Fix: Docker Desktop was reopened; `docker version` showed the server again and the local images were still present.

## 2026-10-19 - Full training ran out of GPU memory at step 1,700 of 3,000

- What happened: pipeline execution 1 failed in the Train step after 27 minutes of training: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.96 GiB` on the 14.56 GiB T4. Validation loss had gone from 2.169 to 0.755 by step 1,500. Billable time 2,039 s, about 0.42 USD, with no usable model. Evaluate and Register did not run.
- Root cause: the 1.96 GiB request matches the float32 logits of one batch at the maximum target length: 16 sentences x 128 tokens x 256,206 vocabulary entries x 4 bytes. Batches are grouped by length, so the longest batch arrived late, when 2.01 GiB of GPU memory was reserved but fragmented. The 30-step smoke run never reached a full-length batch.
- Fix: per-device batch 8 with 4 accumulation steps (same effective batch of 32, half the logits peak), and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the training job environment against fragmentation. The run is not resumed from the step-1,500 checkpoint: changing the per-device batch changes the data order, and a clean run is simpler to report.

## 2026-10-19 - Register step denied CreateModelPackageGroup

- What happened: pipeline execution 2 trained (3,000 steps), evaluated and passed the chrF gate, then the Register step failed in one second: the pipeline role "is not authorized to perform: sagemaker:CreateModelPackageGroup" on the model package group. No model version was created.
- Root cause: the RegisterModel step calls CreateModelPackageGroup on the group even when the group already exists (it is created by Terraform). The pipeline policy only allowed CreateModelPackage, and the offline definition tests cannot check IAM.
- Fix: add sagemaker:CreateModelPackageGroup to the RegisterModelVersion statement, scoped to the same group ARN. The trained model and the evaluation are kept; only the failed step is retried.

## 2026-10-19 - Register retry rejected the registry metadata values

- What happened: after the IAM fix, RetryPipelineExecution re-ran only Register, which failed again: CreateModelPackage rejected CustomerMetadataProperties because two values did not match the API pattern `[\p{L}\p{Z}\p{N}_.:/=+\-@]*` (the dataset value contained commas, the promotion rule contained ">").
- Root cause: the value pattern was not checked when the definition was written; the offline tests only checked step wiring.
- Fix: commas replaced by " - " and ">" by "greater than" in pipeline.tf, plus an offline test of every metadata value against the pattern. Not applied yet: second failure of the same step, so no further retry without approval.
