# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.v1.outputs import DraftTokenIds
from vllm.v1.worker.gpu.async_utils import async_copy_to_np
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch


class DraftTokensHandler:
    def __init__(
        self,
        device: torch.device | None = None,
        copy_to_scheduler: bool = False,
    ):
        self.device = device
        self.copy_stream = torch.cuda.Stream(device)
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self.copy_event = torch.cuda.Event(blocking=True)
        # PP>1: earlier ranks embed draft ids, so the scheduler must see the
        # real tokens (not [-1] placeholders) and send them in SchedulerOutput.
        self.copy_to_scheduler = copy_to_scheduler

        self.req_ids: list[str] = []
        self.draft_tokens_np: np.ndarray | None = None
        self.num_draft_tokens: int = 0

    def set_draft_tokens(
        self, input_batch: InputBatch, draft_tokens: torch.Tensor
    ) -> None:
        self.req_ids = input_batch.req_ids
        self.num_draft_tokens = draft_tokens.shape[1]
        if not input_batch.has_structured_output_reqs and not self.copy_to_scheduler:
            # No draft token validation needs to be performed by
            # the scheduler for this batch.
            self.draft_tokens_np = None
            return

        # For spec decoding + structured outputs, we must transfer the
        # draft tokens back to the scheduler for grammar validation.
        current_stream = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.copy_stream):
            self.draft_tokens_np = async_copy_to_np(draft_tokens)
            # draft_tokens is a temporary allocation on the main stream and read here on
            # copy_stream; without record_stream, the caching allocator may reuse its
            # memory before the async copy executes.
            draft_tokens.record_stream(self.copy_stream)
            self.copy_event.record()

    def get_draft_tokens(self) -> DraftTokenIds | None:
        if self.draft_tokens_np is not None:
            self.copy_event.synchronize()
            draft_token_ids = self.draft_tokens_np.tolist()
        else:
            # This case only happens when async scheduling is disabled.
            draft_token_ids = [[-1] * self.num_draft_tokens for _ in self.req_ids]
        return DraftTokenIds(self.req_ids, draft_token_ids)


def copy_scheduled_draft_tokens_to_gpu(
    draft_token_buf: torch.Tensor,
    req_ids: list[str],
    idx_mapping: torch.Tensor,
    scheduled_drafts: dict[str, list[int]],
    device: torch.device,
) -> None:
    """Scatter scheduler draft ids into the GPU buffer used by combine().

    Under PP the embedder (rank 0) never runs propose(), so leftover GPU
    drafts stay zeros unless we copy the ids from SchedulerOutput.
    """
    n_spec = draft_token_buf.shape[-1]
    packed = np.zeros((len(req_ids), n_spec), dtype=np.int64)
    any_real = False
    for i, req_id in enumerate(req_ids):
        toks = scheduled_drafts.get(req_id) or ()
        if not toks:
            continue
        n = min(len(toks), n_spec)
        packed[i, :n] = toks[:n]
        if not any_real:
            any_real = any(t >= 0 for t in toks[:n])
    if not any_real:
        return
    gpu_packed = async_copy_to_gpu(packed, device=device)
    draft_token_buf[idx_mapping] = gpu_packed


def get_parallel_drafting_token_id(hf_config) -> int:
    """Resolve the mask token id used for parallel drafting slots.

    Checks (in order): `dflash_config.mask_token_id`, top-level `mask_token_id`,
    `dspark_noise_token_id`, `pard_token`, `ptd_token_id`. Raises ValueError if
    none are present.
    """
    dflash_config = getattr(hf_config, "dflash_config", None) or {}
    if "mask_token_id" in dflash_config:
        return int(dflash_config["mask_token_id"])
    if getattr(hf_config, "mask_token_id", None) is not None:
        return int(hf_config.mask_token_id)
    if hasattr(hf_config, "dspark_noise_token_id"):
        return int(hf_config.dspark_noise_token_id)
    if hasattr(hf_config, "pard_token"):
        return int(hf_config.pard_token)
    if hasattr(hf_config, "ptd_token_id"):
        return int(hf_config.ptd_token_id)
    raise ValueError(
        "Model config must specify `dflash_config.mask_token_id`,"
        " `mask_token_id`, `dspark_noise_token_id`, `pard_token`, or"
        " `ptd_token_id` for parallel drafting."
    )
