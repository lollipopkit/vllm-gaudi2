# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.platforms import current_platform

if TYPE_CHECKING:
    from .hpu.loop import (
        Inkling2HPUForConditionalGeneration as Inkling2ForConditionalGeneration,
        Inkling2HPULoopForConditionalGeneration as Inkling2LoopForConditionalGeneration,
    )

__all__ = [
    "Inkling2ForConditionalGeneration",
    "Inkling2LoopForConditionalGeneration",
]


def __getattr__(name: str):
    if name in __all__:
        if current_platform.device_type == "hpu":
            from .hpu import loop as hpu_loop

            mapping = {
                "Inkling2ForConditionalGeneration": hpu_loop.Inkling2HPUForConditionalGeneration,
                "Inkling2LoopForConditionalGeneration": hpu_loop.Inkling2HPULoopForConditionalGeneration,
            }
            return mapping[name]
        raise NotImplementedError(
            f"Inkling2 ({name}) in this repository is currently implemented for HPU (Gaudi2)."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
