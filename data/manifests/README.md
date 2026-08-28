# Frozen evaluation manifests

These files describe the frozen evaluation membership used in the paper
without redistributing BBQ or BBQ-V content.

| File | Contents | Count |
| --- | --- | ---: |
| `bbq_frozen_ids.csv` | Metadata-only IDs for the full BBQ frozen set | 2,000 rows / 500 cases |
| `bbq_v_frozen_ids.csv` | Metadata-only IDs for the full BBQ-V frozen set | 1,998 rows / 389 scenarios |
| `bbq_m4_subsets.json` | Exact nested M4/M5 BBQ unit memberships | metadata only |
| `bbq_v_m4_subsets.json` | Exact nested M4/M5 BBQ-V unit memberships | metadata only |

## Upstream provenance

- **BBQ:** [NYU BBQ](https://github.com/nyu-mll/BBQ), pinned commit
  `bea11bd97d79217245b5871acd247b9d6eb24598`. The frozen full sample has
  fingerprint `69633d39a5318756acb55745bbd606395cbccc1371402f8adc2855a8a34499df`.
- **BBQ-V:** [UCF-CRCV BBQ-Vision](https://github.com/UCF-CRCV/BBQ-Vision),
  pinned repository commit `16408994a0607d673c18ad6331a144fed741f9dc` and
  visual-release revision `a1c78b8f73bc40408993414e3d94714a6a9169d3`. The
  frozen full sample has fingerprint
  `afc11a90267bebd1e726037caf1b44f05cded712cc6c84f033600672b1a16f3b`.

Users must obtain both benchmarks from their original projects and comply with
their original terms. These manifests do not contain questions, answer choices,
labels, annotations, images, or model/judge outputs.

## Identifier and natural-unit rules

### BBQ

`category` plus `example_id` identifies a source row. `case_id` and
`natural_unit_id` make the complete-case requirement explicit: every selected
case must contain exactly four rows covering the two context conditions and
two question polarities. The manifest is in the accepted output order.

### BBQ-V

`id`, `cross_id`, and `unique_question_id` identify a source row from the
pinned visual and official MCQ sources. `unique_question_id` is the complete
scenario natural unit. `image_sha256` verifies correspondence to the pinned
source image but is not an image file or location. The manifest is in the
accepted output order; 999 unique image hashes occur across 1,998 rows.

## Validate a prepared reconstruction

```bash
python scripts/verify_frozen_manifests.py \
  --bbq-sample /path/to/bbq/sample.csv \
  --bbq-v-sample /path/to/bbq_v/sample.csv
```

See [`REPRODUCIBILITY.md`](../../REPRODUCIBILITY.md) for the full procedure.
