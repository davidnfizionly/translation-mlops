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
