#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import builtins
import logging
import math
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from typing_extensions import Unpack

from lerobot.utils.import_utils import _transformers_available

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
else:
    CONFIG_MAPPING = None
    PaliGemmaForConditionalGeneration = None

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OPENPI_ATTENTION_MASK_VALUE,
)

# Import FAST tokenizer
import sys
from pathlib import Path as PathLib

# Add scripts directory to path if needed
# Try to find the scripts directory relative to workspace root
workspace_root = PathLib(__file__).parent.parent.parent.parent.parent.parent.parent
scripts_path = workspace_root / "scripts"
if scripts_path.exists() and str(scripts_path) not in sys.path:
    sys.path.insert(0, str(scripts_path))

try:
    from pi05.fast_tokenizer import FASTTokenizer
except ImportError:
    # Fallback: try direct import if scripts is in path
    import importlib.util
    tokenizer_path = workspace_root / "scripts" / "pi05" / "fast_tokenizer.py"
    if tokenizer_path.exists():
        spec = importlib.util.spec_from_file_location("fast_tokenizer", tokenizer_path)
        fast_tokenizer_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fast_tokenizer_module)
        FASTTokenizer = fast_tokenizer_module.FASTTokenizer
    else:
        raise ImportError(f"Could not find fast_tokenizer.py at {tokenizer_path}")

logger = logging.getLogger(__name__)

PALIGEMMA_EOS_TOKEN = 1


class ActionSelectKwargs(TypedDict, total=False):
    max_decoding_steps: int | None
    temperature: float | None


def make_attn_mask(input_mask: torch.Tensor, mask_ar: torch.Tensor) -> torch.Tensor:
    """Adapted from big_vision (OpenPI implementation).

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    
    Returns:
      attn_mask: bool[B, N, N] 2D attention mask
    """
    # [B, N] -> [B, N] (broadcast if needed)
    mask_ar = mask_ar.broadcast_to(input_mask.shape)
    # [B, N] -> [B, N] cumulative sum
    cumsum = torch.cumsum(mask_ar, dim=1)
    # [B, N, N] attention mask: cumsum[:, None, :] <= cumsum[:, :, None]
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    # [B, N, N] valid mask: only attend to valid (non-padding) tokens
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return attn_mask & valid_mask


def left_to_right_align(
    x: torch.Tensor, input_mask: torch.Tensor, attn_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Converts input from left-align to right-aligned (PyTorch version of OpenPI).

    Args:
      x: [B, N, D] token embeddings
      input_mask: [B, N] padding mask
      attn_mask: [B, N, N] attention mask
    
    Returns:
      Right-aligned x, input_mask, attn_mask
    """
    # Find sequence length for each batch element
    # [B, N] -> [B] max index where mask is True
    seqlen = (input_mask * torch.arange(input_mask.shape[1], device=input_mask.device)).max(dim=1)[0] + 1
    
    # Roll each sequence to the right by (max_len - seqlen)
    # [B, N, D] -> [B, N, D]
    x = torch.stack([torch.roll(x[i], -seqlen[i].item(), dims=0) for i in range(x.shape[0])])
    input_mask = torch.stack([torch.roll(input_mask[i], -seqlen[i].item(), dims=0) for i in range(input_mask.shape[0])])
    attn_mask = torch.stack([torch.roll(attn_mask[i], -seqlen[i].item(), dims=(0, 1)) for i in range(attn_mask.shape[0])])
    
    return x, input_mask, attn_mask


def put_along_last_axis(arr: torch.Tensor, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Like np.put_along_axis(..., axis=-1), PyTorch version of OpenPI.

    Args:
      arr: [B, N] or [B, N, D] array to modify
      indices: [B, 1] or [B, N] indices where to put values
      values: [B, 1] or [B, N] values to put
    
    Returns:
      Modified array
    """
    assert arr.ndim == indices.ndim == values.ndim, (arr.ndim, indices.ndim, values.ndim)
    
    # Create one-hot encoding for indices
    # [B, 1] -> [B, 1, vocab_size] or [B, N] -> [B, N, vocab_size]
    onehot = F.one_hot(indices, arr.shape[-1]).to(dtype=values.dtype)
    
    # Create mask: [B, N] -> [B, N, vocab_size]
    put_mask = torch.einsum("...i,...in->...n", torch.ones_like(values, dtype=torch.int32), onehot)
    
    # Create values to put: [B, N] -> [B, N, vocab_size]
    put_values = torch.einsum("...i,...in->...n", values, onehot)
    
    # Apply: where mask is True, use put_values, else keep arr
    return torch.where(put_mask.bool(), put_values, arr)


def pad_vector(vector, new_dim):
    """Pad the last dimension of a vector to new_dim with zeros.

    Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def resize_with_pad_torch(
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """PyTorch version of resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)

    Returns:
        Resized and padded tensor with same shape format as input
    """
    # Check if input is in channels-last format [*b, h, w, c] or channels-first [*b, c, h, w]
    if images.shape[-1] <= 4:  # Assume channels-last format
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
        images = images.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension

    batch_size, channels, cur_height, cur_width = images.shape

    # Calculate resize ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Resize
    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    # Handle dtype-specific clipping
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Pad
    constant_value = 0 if images.dtype == torch.uint8 else -1.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),  # left, right, top, bottom
        mode="constant",
        value=constant_value,
    )

    # Convert back to original format if needed
    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

    return padded_images


class GemmaConfig:
    """Configuration for Gemma model variants."""

    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_config(variant: str) -> GemmaConfig:
    """Returns config for specified gemma variant."""
    if variant == "gemma_300m":
        return GemmaConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    elif variant == "gemma_2b":
        return GemmaConfig(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


class PI05Fast(nn.Module):
    """Core PI05 Fast model - autoregressive generation (no flow matching)."""

    def __init__(self, config: PI05Config):
        super().__init__()
        self.config = config

        paligemma_config = get_gemma_config(config.paligemma_variant)

        # Create PaliGemma config for HuggingFace
        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = paligemma_config.width
        vlm_config_hf.text_config.intermediate_size = paligemma_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = paligemma_config.num_heads
        vlm_config_hf.text_config.head_dim = paligemma_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = paligemma_config.depth
        vlm_config_hf.text_config.num_key_value_heads = paligemma_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = False
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)

        # Set dtype
        if config.dtype == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif config.dtype == "float32":
            self.to(dtype=torch.float32)
        else:
            raise ValueError(f"Invalid dtype: {config.dtype}")

        # Keep certain params in float32
        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)

        msg = """An incorrect transformer version is used, please create an issue on https://github.com/huggingface/lerobot/issues"""

        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma.language_model.gradient_checkpointing = True
        self.paligemma.vision_tower.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for PI05FastPytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma.language_model.gradient_checkpointing = False
        self.paligemma.vision_tower.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for PI05FastPytorch model")

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def embed_inputs(
        self, images: list[torch.Tensor], img_masks: list[torch.Tensor], tokens: torch.Tensor, masks: torch.Tensor, token_ar_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images and language tokens (OpenPI Fast model style).
        Args:
          images: List of [B, C, H, W] image tensors
          img_masks: List of [B] image mask tensors
          tokens: [B, N] tokenized prompt tokens
          masks: [B, N] token padding masks
          token_ar_mask: [B, N] token autoregressive mask (optional, defaults to all 0s)
        
        Returns:
          token_embeddings: [B, S, D] concatenated embeddings
          input_mask: [B, S] padding mask
          ar_mask: [B, S] autoregressive mask (0=bidirectional, 1=causal)
        """
        input_mask_list = []
        ar_mask_list = []
        token_embeddings_list = []

        # Embed images
        for img, img_mask in zip(images, img_masks, strict=True):
            def image_embed_func(img):
                return self.paligemma.model.get_image_features(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            # img_emb: [B, num_img_tokens, D]
            bsize, num_img_embs = img_emb.shape[:2]

            token_embeddings_list.append(img_emb)
            # Expand mask: [B] -> [B, num_img_tokens]
            input_mask_list.append(img_mask[:, None].expand(bsize, num_img_embs))
            # Image tokens attend to each other --> AR mask = 0 (bidirectional)
            ar_mask_list.append(torch.zeros(bsize, num_img_embs, dtype=torch.long, device=img_emb.device))

        # Embed language tokens
        def lang_embed_func(tokens):
            lang_emb = self.paligemma.language_model.embed_tokens(tokens)
            # Scale by sqrt of embedding dimension (OpenPI convention)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        # lang_emb: [B, N, D]
        token_embeddings_list.append(lang_emb)
        input_mask_list.append(masks)
        # Language tokens AR mask comes from input (0=bidirectional, 1=causal)
        if token_ar_mask is not None:
            ar_mask_list.append(token_ar_mask.to(dtype=torch.long))
        else:
            # Default: all bidirectional
            num_lang_embs = lang_emb.shape[1]
            ar_mask_list.append(torch.zeros(bsize, num_lang_embs, dtype=torch.long, device=lang_emb.device))

        # Concatenate all embeddings
        token_embeddings = torch.cat(token_embeddings_list, dim=1)  # [B, S, D]
        input_mask = torch.cat(input_mask_list, dim=1)  # [B, S]
        ar_mask = torch.cat(ar_mask_list, dim=1)  # [B, S]

        return token_embeddings, input_mask, ar_mask

    def forward(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        tokens: torch.Tensor,
        masks: torch.Tensor,
        token_ar_mask: torch.Tensor,
        token_loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss on tokens using the FAST tokenizer.
        

        Args:
          images: List of [B, C, H, W] image tensors
          img_masks: List of [B] image mask tensors
          tokens: [B, N] tokenized prompt tokens (includes prefix + suffix)
          masks: [B, N] token padding masks
          token_ar_mask: [B, N] autoregressive mask (0=bidirectional, 1=causal)
          token_loss_mask: [B, N] loss mask (True where we compute loss)
        
        Returns:
          loss: [B] per-sample loss
        """
        # Embed inputs: [B, S, D], [B, S], [B, S]
        input_token_embeddings, input_mask, ar_mask = self.embed_inputs(images, img_masks, tokens, masks, token_ar_mask)

        # Create 2D attention mask: [B, S, S]
        attn_mask = make_attn_mask(input_mask, ar_mask)

        # Compute one-hot targets: we predict *next* token, so shift input tokens by one
        # targets: [B, N-1, vocab_size]
        targets = F.one_hot(tokens[:, 1:], num_classes=self.paligemma.config.text_config.vocab_size).float()

        # Each input predicts *next* token, so we don't input the last token
        # input_token_embeddings[:, :-1]: [B, S-1, D]
        # attn_mask[:, :-1, :-1]: [B, S-1, S-1]
        attn_mask_4d = self._prepare_attention_masks_4d(attn_mask[:, :-1, :-1])
        position_ids = torch.cumsum(input_mask[:, :-1], dim=1) - 1

        def forward_func(input_embeds, attn_mask_4d, position_ids):
            output = self.paligemma.language_model.forward(
                inputs_embeds=input_embeds,
                attention_mask=attn_mask_4d,
                position_ids=position_ids,
                use_cache=False,
            )
            return output.last_hidden_state

        # pre_logits: [B, S-1, D]
        pre_logits = self._apply_checkpoint(forward_func, input_token_embeddings[:, :-1], attn_mask_4d, position_ids)

        # Only decode logits for the target tokens to save memory
        # Decode only the last N-1 positions (matching targets)
        # pre_logits[:, -targets.shape[1]:]: [B, N-1, D]
        logits = self.paligemma.language_model.lm_head(pre_logits[:, -targets.shape[1]:])
        # logits: [B, N-1, vocab_size]
        logp = F.log_softmax(logits, dim=-1)

        # Compute CE loss on token targets
        # loss_mask: [B, N] -> [B, N-1] (shifted)
        loss_mask = token_loss_mask[:, 1:]
        # token_pplx: [B, N-1] per-token perplexity
        token_pplx = torch.sum(targets * logp, dim=-1)
        # loss: [B] per-sample loss
        loss = -torch.sum(token_pplx * loss_mask, dim=-1) / torch.clamp(torch.sum(loss_mask, dim=-1), min=1)

        return loss

    @torch.no_grad()
    def sample_actions(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        tokens: torch.Tensor,
        masks: torch.Tensor,
        token_ar_mask: torch.Tensor,
        max_decoding_steps: int = 256,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        """Sample actions autoregressively (OpenPI Fast model style).

        Args:
          images: List of [B, C, H, W] image tensors
          img_masks: List of [B] image mask tensors
          tokens: [B, N] tokenized prompt tokens (prefix only)
          masks: [B, N] token padding masks
          token_ar_mask: [B, N] autoregressive mask
          max_decoding_steps: Maximum number of decoding steps
          temperature: Sampling temperature (0.0 = greedy)

        Returns:
          output_tokens: [B, max_decoding_steps] generated tokens
        """
        # Embed inputs
        prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_inputs(images, img_masks, tokens, masks, token_ar_mask)

        # Create attention mask
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)

        # Left to right align all input token sequences
        prefix_token_embeddings, prefix_mask, prefix_attn_mask = left_to_right_align(
            prefix_token_embeddings, prefix_mask, prefix_attn_mask
        )
        
        prefill_size = prefix_token_embeddings.shape[1]
        prefill_len = torch.sum(prefix_mask, dim=-1)  # [B]
        prefix_start = prefill_size - prefill_len  # [B]

        # First fill KV cache with a forward pass of the prefix
        # Pad attention mask to set the size of the KV cache (prefill_size + max_decoding_steps)
        prefix_attn_mask_padded = F.pad(prefix_attn_mask, (0, max_decoding_steps, 0, 0, 0, 0))
        prefix_position_ids = torch.cumsum(prefix_mask, dim=-1) - 1
        prefix_attn_mask_4d = self._prepare_attention_masks_4d(prefix_attn_mask_padded)
        
        self.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        
        prefix_output = self.paligemma.language_model.forward(
            inputs_embeds=prefix_token_embeddings,
            attention_mask=prefix_attn_mask_4d,
            position_ids=prefix_position_ids,
            use_cache=True,
        )
        prefix_logits = self.paligemma.language_model.lm_head(prefix_output.last_hidden_state)
        kv_cache = prefix_output.past_key_values

        # Prepare decoding -- final logit decodes the first token
        last_logit = prefix_logits[:, -1:]  # [B, 1, vocab_size]
        output_tokens = torch.zeros((last_logit.shape[0], max_decoding_steps), dtype=torch.long, device=tokens.device)

        # Autoregressive decoding loop
        for step in range(max_decoding_steps):
            # Sample token from last logit
            if temperature > 0.0:
                probs = F.softmax(last_logit.squeeze(1) / temperature, dim=-1)  # [B, vocab_size]
                token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            else:
                token = torch.argmax(last_logit.squeeze(1), dim=-1, keepdim=True)  # [B, 1]

            # Store token
            output_tokens[:, step] = token.squeeze(-1)

            # Check for early stopping --> stop if all batch elements have EOS token
            has_eos = (token.squeeze(-1) == PALIGEMMA_EOS_TOKEN)  # [B]
            if has_eos.all():
                break

            # Decode one step
            token_embedding = self.paligemma.language_model.embed_tokens(token)  # [B, 1, D]
            positions = (prefill_len[:, None] + step + 1)  # [B, 1]
            
            # Create attention mask for this step
            # Token can attend to all previous tokens (from prefix_start onwards)
            seq_len = prefill_size + max_decoding_steps
            mask = (
                (torch.arange(seq_len, device=token.device)[None, None, :] >= prefix_start[:, None, None])
                & (torch.arange(seq_len, device=token.device)[None, None, :] < (prefill_size + step + 1)[:, None, None])
            )
            mask_4d = self._prepare_attention_masks_4d(mask)

            output = self.paligemma.language_model.forward(
                inputs_embeds=token_embedding,
                attention_mask=mask_4d,
                position_ids=positions,
                past_key_values=kv_cache,
                use_cache=True,
            )
            last_logit = self.paligemma.language_model.lm_head(output.last_hidden_state)  # [B, 1, vocab_size]
            kv_cache = output.past_key_values

        return output_tokens


class PI05FastPolicy(PreTrainedPolicy):
    """PI05 Fast Policy for LeRobot - autoregressive generation."""

    config_class = PI05Config
    name = "pi05_fast"

    def __init__(
        self,
        config: PI05Config,
    ):
        """
        Args:
            config: Policy configuration class instance.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Initialize FAST tokenizer
        self.fast_tokenizer = FASTTokenizer(
            max_len=config.tokenizer_max_length,
            fast_tokenizer_path=getattr(config, "fast_tokenizer_path", "physical-intelligence/fast"),
            fast_skip_tokens=getattr(config, "fast_skip_tokens", 0),
        )

        # Initialize the core PI05 Fast model
        self.model = PI05Fast(config)

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)

        self.reset()

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """Override the from_pretrained method."""
        print(
            "The PI05 Fast model is a direct port of the OpenPI PI0 Fast implementation. \n"
            "This implementation follows the original OpenPI structure for compatibility. \n"
            "Original implementation: https://github.com/Physical-Intelligence/openpi"
        )
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        # Use provided config if available, otherwise create default config
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        # Initialize model without loading weights
        model = cls(config, **kwargs)

        # Load weights if available
        try:
            print(f"Loading model from: {pretrained_name_or_path}")
            try:
                from transformers.utils import cached_file

                # Try safetensors first
                resolved_file = cached_file(
                    pretrained_name_or_path,
                    "model.safetensors",
                    cache_dir=kwargs.get("cache_dir"),
                    force_download=kwargs.get("force_download", False),
                    resume_download=kwargs.get("resume_download"),
                    proxies=kwargs.get("proxies"),
                    use_auth_token=kwargs.get("use_auth_token"),
                    revision=kwargs.get("revision"),
                    local_files_only=kwargs.get("local_files_only", False),
                )
                from safetensors.torch import load_file

                original_state_dict = load_file(resolved_file)
                print("✓ Loaded state dict from model.safetensors")
            except Exception as e:
                print(f"Could not load state dict from remote files: {e}")
                print("Returning model without loading pretrained weights")
                return model

            # Remap state dict keys
            remapped_state_dict = {}
            for key, value in original_state_dict.items():
                # Add "model." prefix if not present
                if not key.startswith("model."):
                    new_key = f"model.{key}"
                    remapped_state_dict[new_key] = value
                else:
                    remapped_state_dict[key] = value

            # Load the remapped state dict
            missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=strict)

            if missing_keys:
                print(f"Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"Unexpected keys: {len(unexpected_keys)}")
            if not missing_keys and not unexpected_keys:
                print("All keys loaded successfully!")

        except Exception as e:
            print(f"Warning: Could not load state dict: {e}")

        return model

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        """Reset internal state - called when environment resets."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model.

        Images from LeRobot are typically in [B, C, H, W] format and normalized to [0, 1].
        PaliGemma expects images in [B, C, H, W] format and normalized to [-1, 1].
        """
        images = []
        img_masks = []

        # Get device from model parameters
        device = next(self.parameters()).device

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]

            # Ensure tensor is on the same device as the model
            if img.device != device:
                img = img.to(device)

            # Ensure float32 dtype for consistency
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            # Handle both [B, C, H, W] and [B, H, W, C] formats
            is_channels_first = img.shape[1] == 3

            if is_channels_first:
                # Convert [B, C, H, W] to [B, H, W, C] for processing
                img = img.permute(0, 2, 3, 1)

            # Resize with padding if needed
            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)

            # Normalize from [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0

            # Convert back to [B, C, H, W] format if it was originally channels-first
            if is_channels_first:
                img = img.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

            images.append(img)
            # Create mask (all ones for real images)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)

        # Create image features not present in the batch as fully 0 padded images
        for _num_empty_cameras in range(len(missing_img_keys)):
            img = torch.ones_like(img) * -1  # Padded with -1 for SigLIP
            mask = torch.zeros_like(mask)  # Mask is zero for empty cameras
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        self.eval()

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # Transpose to get shape (n_action_steps, batch_size, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        
        # Get state from batch
        state = batch.get("observation.state", None)
        if state is None:
            raise ValueError("observation.state is required for PI05 Fast model")
        
        # Get prompt from batch (language tokens)
        # For Fast model, we need to tokenize prompt + state
        prompt = batch.get("observation.language", None)
        batch_size = state.shape[0] if state.ndim > 1 else 1
        
        if prompt is None:
            # Try to get from language tokens directly
            tokens = batch.get(f"{OBS_LANGUAGE_TOKENS}", None)
            masks = batch.get(f"{OBS_LANGUAGE_ATTENTION_MASK}", None)
            token_ar_mask = batch.get("observation.token_ar_mask", None)
            if tokens is None:
                raise ValueError("observation.language or observation.language_tokens is required")
        else:
            # Tokenize using FAST tokenizer for each batch element
            tokens_list = []
            masks_list = []
            token_ar_mask_list = []
            for i in range(batch_size):
                prompt_i = prompt[i] if isinstance(prompt, (list, tuple)) else (prompt if batch_size == 1 else prompt[i])
                state_i = state[i] if state.ndim > 1 else state
                tok, mask, ar_mask, _ = self.fast_tokenizer.tokenize(prompt_i, state_i)
                tokens_list.append(tok)
                masks_list.append(mask)
                token_ar_mask_list.append(ar_mask)
            tokens = torch.stack(tokens_list)
            masks = torch.stack(masks_list)
            token_ar_mask = torch.stack(token_ar_mask_list)

        # Sample tokens using the model
        max_decoding_steps = kwargs.get("max_decoding_steps", 256)
        temperature = kwargs.get("temperature", 0.0)
        
        output_tokens = self.model.sample_actions(
            images, img_masks, tokens, masks, token_ar_mask, max_decoding_steps=max_decoding_steps, temperature=temperature
        )

        # Extract actions from tokens using FAST tokenizer
        # output_tokens: [B, max_decoding_steps]
        actions_list = []
        for i in range(output_tokens.shape[0]):
            # Combine prefix and generated tokens
            full_tokens = torch.cat([tokens[i], output_tokens[i]], dim=0)
            actions = self.fast_tokenizer.extract_actions(
                full_tokens, self.config.chunk_size, self.config.max_action_dim
            )
            actions_list.append(actions)

        # Stack and reshape: [B, chunk_size, action_dim]
        actions = torch.stack(actions_list)
        
        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training."""
        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        
        # Get state and actions from batch
        state = batch.get("observation.state", None)
        if state is None:
            raise ValueError("observation.state is required for PI05 Fast model")
        
        actions = self.prepare_action(batch)
        
        # Get prompt
        prompt = batch.get("observation.language", None)
        batch_size = state.shape[0] if state.ndim > 1 else 1
        
        if prompt is None:
            tokens = batch.get(f"{OBS_LANGUAGE_TOKENS}", None)
            masks = batch.get(f"{OBS_LANGUAGE_ATTENTION_MASK}", None)
            token_ar_mask = batch.get("observation.token_ar_mask", None)
            token_loss_mask = batch.get("observation.token_loss_mask", None)
            if tokens is None:
                raise ValueError("observation.language or observation.language_tokens is required")
        else:
            # Tokenize using FAST tokenizer for each batch element
            tokens_list = []
            masks_list = []
            token_ar_mask_list = []
            token_loss_mask_list = []
            for i in range(batch_size):
                prompt_i = prompt[i] if isinstance(prompt, (list, tuple)) else (prompt if batch_size == 1 else prompt[i])
                state_i = state[i] if state.ndim > 1 else state
                actions_i = actions[i] if actions.ndim > 2 else actions
                tok, mask, ar_mask, loss_mask = self.fast_tokenizer.tokenize(prompt_i, state_i, actions_i)
                tokens_list.append(tok)
                masks_list.append(mask)
                token_ar_mask_list.append(ar_mask)
                token_loss_mask_list.append(loss_mask)
            tokens = torch.stack(tokens_list)
            masks = torch.stack(masks_list)
            token_ar_mask = torch.stack(token_ar_mask_list)
            token_loss_mask = torch.stack(token_loss_mask_list)

        # Compute loss
        losses = self.model.forward(images, img_masks, tokens, masks, token_ar_mask, token_loss_mask)

        loss = losses.mean()

        loss_dict = {
            "loss": loss.item(),
            "loss_per_sample": losses.detach().cpu().numpy().tolist(),
        }

        return loss, loss_dict
