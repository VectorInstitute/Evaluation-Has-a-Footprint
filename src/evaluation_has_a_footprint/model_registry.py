"""Finalized model metadata, with no model-loading dependencies."""

from __future__ import annotations

from types import MappingProxyType


MODELS = MappingProxyType(
    {
        "qwen25_vl_7b": {
            "hf_id": "Qwen/Qwen2.5-VL-7B-Instruct",
            "revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
            "family": "dense_multimodal",
            "modalities": ("text", "image_text"),
        },
        "qwen3_vl_30b_a3b": {
            "hf_id": "Qwen/Qwen3-VL-30B-A3B-Instruct",
            "revision": "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c",
            "family": "multimodal_moe",
            "modalities": ("text", "image_text"),
        },
    }
)
