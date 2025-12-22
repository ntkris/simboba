"""Prompt templates for simboba."""

from simboba.prompts.generation import (
    DATASET_GENERATION_PROMPT,
    build_dataset_generation_prompt,
)
from simboba.prompts.judge import (
    JUDGE_PROMPT,
    build_judge_prompt,
    format_conversation,
)

__all__ = [
    "DATASET_GENERATION_PROMPT",
    "build_dataset_generation_prompt",
    "JUDGE_PROMPT",
    "build_judge_prompt",
    "format_conversation",
]
