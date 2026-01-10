import logging
from typing import Optional

import numpy as np
import torch
from transformers import AutoProcessor


class FASTTokenizer:
    def __init__(self, max_len: int = 256,
                 fast_tokenizer_path: str = "physical-intelligence/fast",
                 fast_skip_tokens: int = 0) -> None:
        self._max_len = max_len
        self._fast_skip_tokens = fast_skip_tokens

        self._fast_tokenizer = AutoProcessor.from_pretrained(
            fast_tokenizer_path,
            trust_remote_code=True,
        )

    def tokenize(
        self,
        prompt: str,
        state: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize a prompt, state, and actions."""
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(state, torch.Tensor):
            state = [state]
        if actions is not None:
            if isinstance(actions, torch.Tensor):
                actions = [actions]
            else:
                actions = [torch.tensor(actions)]
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convert state to numpy for digitize operation, then back to tensor
        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        state_np = state.detach().cpu().numpy() if isinstance(state, torch.Tensor) else np.asarray(state)
        discretized_state = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._fast_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            # Convert actions to numpy for FAST tokenizer if needed
            actions_np = actions.detach().cpu().numpy() if isinstance(actions, torch.Tensor) else np.asarray(actions)
            # Tokenize actions with FAST tokenizer --> map to last tokens in PaliGemma vocab
            action_tokens = self._fast_tokenizer(actions_np[None])[0]
            action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(action_tokens)

            # Convention: postfix contains 'Action:' followed by FAST tokens, followed by '|'
            postfix_tokens = (
                self._fast_tokenizer.encode("Action: ")
                + action_tokens_in_pg.tolist()
                + self._fast_tokenizer.encode("|", add_eos=True)
            )
        else:
            postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [0] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + [False] * (self._max_len - tokens_len)
            ar_mask = ar_mask + [0] * (self._max_len - tokens_len)
            loss_mask = loss_mask + [False] * (self._max_len - tokens_len)
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        # Convert to PyTorch tensors
        return (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(token_mask, dtype=torch.bool),
            torch.tensor(ar_mask, dtype=torch.long),
            torch.tensor(loss_mask, dtype=torch.bool),
        )

    def tokenize_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cleaned_text = prompt.lower().strip().replace("_", " ")
        text_tokens = self._fast_tokenizer.encode(cleaned_text, add_bos=True)
        token_mask = [True] * len(text_tokens)
        ar_mask = [0] * len(text_tokens)
        loss_mask = [False] * len(text_tokens)

        tokens_len = len(text_tokens)
        if tokens_len < self._max_len:
            padding = [0] * (self._max_len - tokens_len)
            text_tokens = text_tokens + padding
            token_mask = token_mask + [False] * (self._max_len - tokens_len)
            ar_mask = ar_mask + [0] * (self._max_len - tokens_len)
            loss_mask = loss_mask + [False] * (self._max_len - tokens_len)
        else:
            assert False

        # Convert to PyTorch tensors
        return (
            torch.tensor(text_tokens, dtype=torch.long),
            torch.tensor(token_mask, dtype=torch.bool),
            torch.tensor(ar_mask, dtype=torch.long),
            torch.tensor(loss_mask, dtype=torch.bool),
        )

    def extract_actions(self, tokens: torch.Tensor, action_horizon: int, action_dim: int) -> torch.Tensor:
        # Convert tokens to list for decoding
        tokens_list = tokens.detach().cpu().tolist() if isinstance(tokens, torch.Tensor) else tokens.tolist()
        # Decode predicted output tokens
        decoded_tokens = self._fast_tokenizer.decode(tokens_list)

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return torch.zeros((action_horizon, action_dim), dtype=torch.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._fast_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        )
        # The mapping is symmetric, so we can use the same function to reverse it
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        actions_np = self._fast_tokenizer.decode(
            [action_tokens.tolist()], time_horizon=action_horizon, action_dim=action_dim
        )[0]
        return torch.from_numpy(actions_np).float()

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int] | torch.Tensor) -> np.ndarray:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().cpu().numpy()
        elif isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._fast_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens
