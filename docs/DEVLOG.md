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
