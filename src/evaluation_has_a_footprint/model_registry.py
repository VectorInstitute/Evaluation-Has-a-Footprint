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
        "gemma4_12b": {
            "hf_id": "google/gemma-4-12B-it",
            "revision": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
            "family": "gemma4_unified",
            "modalities": ("text", "image_text"),
        },
    }
)
