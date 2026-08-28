# Data and licenses

This repository does not redistribute benchmark questions, answer choices,
labels, images, model weights, or the internal experiment archive. Obtain and
prepare all upstream data separately, and follow the upstream terms below.

## BBQ

[BBQ: A Hand-Built Bias Benchmark for Question Answering](https://github.com/nyu-mll/BBQ)
is the text benchmark used here. Obtain its data from the upstream repository
and prepare it locally before using this runner. The upstream repository is
licensed under [CC BY 4.0](https://github.com/nyu-mll/BBQ/blob/main/LICENSE);
cite the BBQ paper and comply with that license when using the dataset.

## BBQ-V

[BBQ-V: Benchmarking Visual Stereotype Bias in Large Multimodal Models](https://github.com/UCF-CRCV/BBQ-Vision)
is the image-and-text benchmark used here. Obtain it from the official project
under its original access terms, then prepare it locally before using this
runner. Its official repository and dataset documentation state
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) and
academic-research use for the images; users are responsible for complying with
those terms and any applicable source-image requirements.

## Frozen membership metadata

`data/manifests/` contains only membership IDs and provenance/validation
metadata. It contains no benchmark questions, answers, answer options, labels,
images, predictions, or judge artifacts. The manifests identify the frozen
evaluation membership used here and must be used with separately prepared
upstream BBQ and BBQ-V data. They do not grant rights to either dataset.

## Models

The public portable runner includes the following Qwen model cards and
immutable revisions. Model weights are not redistributed here; obtain them
from their original sources and comply with their upstream licenses.

- [`Qwen/Qwen2.5-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct),
  revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`.
- [`Qwen/Qwen3-VL-30B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct),
  revision `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`.

The paper also reports Gemma-4-12B-it results; no Gemma model weights or
runnable Gemma profile are distributed by this repository.
