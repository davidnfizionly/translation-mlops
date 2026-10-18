# Architecture Decision Records

One line each: what, why, what was rejected.

- **ADR-001 Dataset under CC BY-NC 4.0 accepted.** The English-Nuer-Dinka parallel corpus by Tajnaam is the only large human-oriented corpus found for both languages; this is a non-commercial portfolio project. Rejected: the MIT-licensed health dataset from the same publisher, because all its Nuer and Dinka text is Google Translate output (stated on its card).
- **ADR-002 No dataset, weights or fine-tuned model published.** Both the dataset and NLLB-200 are CC BY-NC 4.0; the repository holds a download script and a few attributed sample pairs, the fine-tuned weights stay in a private S3 bucket and are deleted at cleanup. Rejected: publishing the fine-tuned model on Hugging Face.
- **ADR-003 Widget runs locally behind a local proxy, no public API.** Keeps the endpoint private and avoids offering a translation service built on non-commercial data and weights. Rejected: API Gateway or a public website.
- **ADR-004 Backend stack keeps local state.** The state bucket cannot hold its own state before it exists; `infra/backend` uses a local, git-ignored state file, and every other stack uses the S3 backend with `use_lockfile`. Rejected: a DynamoDB lock table (no longer needed with S3 native locking).
- **ADR-005 State bucket name gets a random suffix and is passed through a git-ignored `backend.hcl`.** Avoids an account ID in a bucket name and in public files. Rejected: account ID in the bucket name.
- **ADR-006 The budget lives in the backend stack and is created first.** The bucket `depends_on` the budget so the budget exists before any other resource; no notifications, as requested; it covers the whole account (credits and refunds excluded), because a tag filter would need the Project cost allocation tag activated first. Rejected: a separate budget stack (one more state to manage for one resource).
- **ADR-007 Region us-east-1.** EC2 access is already validated in this account and the SageMaker GPU quotas needed exist there. Rejected: other regions.
