# MLOps Translation Pipeline for Underrepresented Languages - Fine-Tuning, Deployment and Monitoring on AWS

---

## Project Description

Dinka and Nuer, spoken by millions of people in South Sudan, are almost absent from modern digital tools. In 2026 an open parallel corpus of roughly 500,000 English-Nuer-Dinka sentence pairs was published on Hugging Face by Tajnaam under the Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0) license. The base model NLLB-200 is also under CC BY-NC 4.0. The dataset and the model weights are not redistributed: the repository holds a download script and a few attributed sample pairs, and the fine-tuned model is not published because of the license. This project does not create that dataset and does not claim to build a production translation system. It builds the infrastructure around the dataset: ingestion, cleaning, fine-tuning, evaluation, a model registry, deployment behind an endpoint, monitoring and per-request cost tracking.

The pipeline is provisioned with **Terraform** and **Amazon SageMaker**. It fine-tunes an existing open multilingual translation model (NLLB-200) on the corpus, compares the result with the base model on a held-out test set using objective metrics (BLEU and chrF), registers the model version, deploys it behind a real-time endpoint and records latency, errors and estimated cost per request.

The translation quality is reported exactly as measured. A single fine-tuning run on a subset of the corpus is not expected to give production-grade translation, and the README says so.

---

## Technologies and Services Used

| Layer | Technology | Purpose |
| --- | --- | --- |
| IaC | **Terraform** | S3 buckets, IAM roles per pipeline stage, Glue, Athena, monitoring; the endpoint in its own stack so it can be created and destroyed alone |
| Data lake | **Amazon S3** | Raw and processed corpus storage |
| ETL and cataloging | **AWS Glue** | Cleaning, train, validation and test split, data catalog |
| Data exploration | **Amazon Athena** | SQL data quality checks before training |
| Fine-tuning | **Amazon SageMaker Training Jobs** (Hugging Face container) | Specializes a pre-trained multilingual model (NLLB-200) on the corpus |
| Model versioning | **SageMaker Model Registry** | Records which model version, trained on which data snapshot, with which hyperparameters, is deployed |
| Orchestration | **SageMaker Pipelines** | Train, evaluate, gate, register as a repeatable flow |
| Inference | **SageMaker real-time endpoint** | Translation serving on the smallest instance that works |
| Quality evaluation | **Python (sacreBLEU: BLEU and chrF)** | Objective comparison of base and fine-tuned models on a held-out test set |
| Monitoring | **Amazon CloudWatch** and custom cost tracking | Latency, errors, drift signals, estimated cost per request |
| Security | **IAM**, scoped per pipeline stage | Least privilege, no wildcard admin |
| Frontend | **React, Vite, Tailwind** (local) | Live translation widget with a local proxy that uses the CLI credentials |

---

## Architecture Patterns Demonstrated

- **Transfer learning instead of training from scratch** - fine-tune an existing multilingual model.
- **Model lineage** - each registered version points to its data snapshot, hyperparameters and metrics.
- **Objective evaluation gate** - the pipeline computes BLEU and chrF and promotes a new version only if it improves on the baseline on the held-out test set.
- **No train and test leakage** - the split is deduplicated, and the check is part of the data pipeline.
- **Drift-aware monitoring** - signals such as input length distribution, output length ratio and latency are tracked, not only whether the endpoint is up.
- **Cost-instrumented inference** - estimated cost per request is tracked and compared with the real bill from Cost Explorer.
- **Endpoint lifetime discipline** - the endpoint exists only during the demo session.

---

## Evaluation and Demo Scenarios

| Scenario | What Happens | What It Proves |
| --- | --- | --- |
| **Before and after fine-tuning** | The same held-out sentences translated by the base model and the fine-tuned model, with BLEU and chrF side by side, and sample outputs including failures | The measured effect of fine-tuning, whatever its size |
| **Live translation** | A public-health style message ("Boil water before drinking") translated through the real endpoint | The pipeline output works end to end |
| **Cost per request** | A running estimated cost counter while requests are sent, compared afterwards with the bill | Unit economics are visible, including idle time |

---

## UI Details

- **Live translation widget** - type or select an English sentence, get the output.
- **Side-by-side view** - base model and fine-tuned model outputs for the same input, with the metrics from the evaluation.
- **Cost counter** - running estimated cost, labeled as an estimate.
- **Attribution banner** - dataset creator, license and link, and the base model license, shown in the UI and in the README.

---

## Use Case Simulation

| Role | What They Would See or Do |
| --- | --- |
| **End user (health worker, educator)** | Types a message and gets a translation |
| **ML or platform engineer** | Reviews the Model Registry to see which version is deployed and why it was promoted |
| **FinOps stakeholder** | Watches the estimated cost per request and the endpoint idle cost |
| **Data governance reviewer** | Checks that dataset attribution and licenses are documented |

---

## Deployment Strategy

SageMaker training jobs and endpoints bill by the hour. Everything is prepared and checked offline first, and every GPU or endpoint launch has a cost estimate and an approval.

1. **Terraform backend and budget**
2. **Dataset verification and offline data preparation** - license, size, fields, duplicates, personal data scan, split
3. **Data stack** - S3, Glue, Athena, with SQL quality checks
4. **Training code, smoke run and baseline evaluation**
5. **Fine-tuning through the SageMaker Pipeline** - with the evaluation gate and the registry
6. **Endpoint, monitoring, cost tracking, widget and scenarios**
7. **Clean-up** - endpoint first, then everything else, verified
8. **Documentation** - architecture diagram and README, written offline

---

## Considered and Rejected

- **Training a model from scratch** - far too costly for the data and the budget, and not how custom models are usually built.
- **A raw EC2 training setup** - SageMaker provides the managed training, registry and endpoint pieces this project is about.
- **Subjective evaluation** - a promotion gate needs a repeatable metric.
- **Multi-account governance, multi-region** - not part of this problem.

---

## Architecture Decisions Worth Documenting

- **Fine-tune NLLB-200 instead of training from scratch**: transfer learning fits the data volume and the budget.
- **SageMaker instead of self-managed training**: managed training, registry and endpoints.
- **BLEU and chrF on a held-out test set instead of subjective review**: a repeatable promotion gate; limitations of the metrics for low-resource languages are stated.
- **Subset of the corpus for the fine-tuning run**: chosen from the cost estimate, with the size recorded.
- **Dataset and base model credited with their licenses**: the infrastructure is the contribution, not the data or the model.
- **Fine-tuned model not published**: the dataset and the base model are both CC BY-NC 4.0; the weights stay in the private S3 bucket and are deleted at cleanup. The widget runs locally with no public service.
- **Quality reported as measured**: a single fine-tune will not give production-grade translation.
