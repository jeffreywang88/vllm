# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frames rendered from the per-stream template must be byte-identical to the
frames the response models would have produced."""

import json

import pytest

from vllm.entrypoints.openai.completion.protocol import (
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
)
from vllm.entrypoints.openai.completion.serving import _delta_frame_template

CREATED = 1706000000


def model_frame(
    request_id: str, model_name: str, index: int, text: str, num_tokens: int
) -> str:
    chunk = CompletionStreamResponse(
        id=request_id,
        object="text_completion",
        created=CREATED,
        model=model_name,
        choices=[
            CompletionResponseStreamChoice(
                index=index,
                text=text,
                logprobs=None,
                finish_reason=None,
                stop_reason=None,
                prompt_token_ids=None,
                token_ids=None,
                weight_versions=None,
                num_tokens=num_tokens,
            )
        ],
        kv_transfer_params=None,
    )
    return f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Hello, world",
        ' "quoted" and \\backslashed\\ ',
        "line\nbreaks\r\ttabs",
        "\x00\x01\x1f\x7f",
        "unicode é 日本語 🚀 \U0001d549\U0010ffff",
    ],
)
@pytest.mark.parametrize("num_tokens", [0, 1, 53])
@pytest.mark.parametrize(
    ("request_id", "model_name", "index"),
    [
        ("cmpl-abc123", "meta-llama/Llama-3.1-8B", 0),
        ('id "with" quotes\\', "модель 🎯", 7),
    ],
)
def test_template_frame_matches_model_frame(
    request_id: str, model_name: str, index: int, text: str, num_tokens: int
):
    template = _delta_frame_template(request_id, CREATED, model_name)
    assert template is not None
    prefix, infix, middle, suffix = template
    encoded = json.dumps(text, ensure_ascii=False)
    frame = f"{prefix}{index}{infix}{encoded}{middle}{num_tokens}{suffix}"
    assert frame == model_frame(request_id, model_name, index, text, num_tokens)


def test_template_is_none_when_sentinels_are_ambiguous():
    assert _delta_frame_template("cmpl-987654321", CREATED, "m") is None
