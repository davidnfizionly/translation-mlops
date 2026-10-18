# Data Cleaning and Split

Source: English-Nuer-Dinka Parallel Corpus by Tajnaam, CC BY-NC 4.0, Hugging Face revision `28db28e6b554975aad147c85655ddd82a4cb4ba4`, 499,880 rows with the fields `english`, `nuer`, `din`.

The Nuer and Dinka references are very probably machine generated (see `proof/02-dataset-facts.txt`); this is not confirmed by the dataset author. Cleaning removes broken rows; it cannot turn the references into human translations.

The same code (`data_prep/corpus.py`, standard library only) runs locally, in the tests and in the Glue Python shell job.

## Rules and rows removed

Rules run in this order. A removed row is counted under the first rule that removes it.

| Order | Rule | Definition | Rows removed |
| --- | --- | --- | --- |
| 0 | normalize | Unicode NFC, zero-width characters (U+200B, U+200C, U+200D, U+2060, U+FEFF) removed, whitespace collapsed and trimmed. Rows are modified, not removed. | 0 (3,250 rows modified) |
| 1 | replacement_char | A field contains U+FFFD, a sign of broken encoding | 31 |
| 2 | empty_field | A field is empty after normalisation | 0 |
| 3 | english_length | English has fewer than 3 or more than 100 words | 113 |
| 4 | target_loop | Nuer or Dinka repeats a word trigram 3 or more times while the English does not | 40,820 |
| 5 | length_ratio | Nuer/English or Dinka/English character ratio outside 0.5 to 2.5 | 1,733 |
| 6 | duplicate_english | English already seen (key: NFC, casefold, punctuation removed) | 3 |
| 7 | duplicate_target | Nuer or Dinka already seen (same key) | 74 |
| | Total removed | | 42,774 (8.6%) |
| | Kept | | 457,106 |

Duplicates keep the first occurrence in file order.

## Split

- Rows are grouped by the English key with every number replaced by 0, so sentences that differ only in numbers share a group and a split (452,743 groups).
- Groups are ordered by SHA-256 of `seed:key` (seed 13). The first groups fill the test split (2,000 rows), the next ones the validation split (2,000 rows), the rest is training. No random state is involved, so the split is reproducible.
- Test and validation rows whose word-trigram Jaccard similarity with any training row is 0.8 or more, on any of the three fields, are removed from evaluation: 34 test rows and 61 validation rows.

| Split | Rows |
| --- | --- |
| train | 453,106 |
| valid | 1,939 |
| test | 1,966 |

The test split is held out before any training and is never used for tuning.

## Leakage check

`data_prep/leakage_check.py` re-reads the three files and counts, for test-train, valid-train and test-valid: exact matches of the normalised key per field, matches of the digit-masked English key, and near duplicates (trigram Jaccard 0.8 or more) per field. It exits with an error if any count is not zero. The Glue job runs the same check and fails on a leak. Output: `proof/05-leakage-check.txt`.

## Known limits

- The loop rule only catches repetition of a word trigram. A wrong but fluent translation passes every rule.
- Template sentences (for example "X is a species of moth in the family Noctuidae") appear in all splits with different names; they are not near duplicates under the 0.8 threshold and are kept.
