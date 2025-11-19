"""
Quick script to play with the π₀ (pi0) policy on a dataset,
without needing a real robot.

It loads:
- the NONHUMAN-RESEARCH/sarm_dataset_aloha_mobile_wash_pan dataset
- the lerobot/pi0_base policy

and runs offline inference on a few frames from the first episode.

Usage (from the repo root, after installing lerobot with `[pi]` extras):

    python play_pi.py
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors


REPO_ID = "NONHUMAN-RESEARCH/sarm_dataset_aloha_mobile_wash_pan"
PI0_MODEL_ID = "lerobot/pi0_base"


def select_device() -> torch.device:
    """Select a reasonable default device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_batch_from_frame(frame: Dict[str, Any], task: str, index: int) -> Dict[str, Any]:
    """
    Convert a single dataset frame into a batch dict expected by the
    PolicyProcessorPipeline.

    - Keeps only observation.* keys as observations
    - Adds a natural-language task string for the π tokenizer
    - Adds an index for convenience / debugging
    """
    batch: Dict[str, Any] = {
        key: value for key, value in frame.items() if key.startswith("observation.")
    }
    batch["task"] = task
    batch["index"] = index
    return batch


def main() -> None:
    device = select_device()
    print(f"Using device: {device}")

    # 1) Load dataset metadata (features, stats, episodes info)
    print(f"Loading dataset metadata for: {REPO_ID}")
    ds_meta = LeRobotDatasetMetadata(REPO_ID)
    print(f"- Available feature keys: {list(ds_meta.features.keys())[:10]} ...")
    print(f"- Number of episodes: {len(ds_meta.episodes)}")

    # 2) Build π₀ config and policy (pretrained weights from the Hub)
    print(f"Loading π₀ policy from: {PI0_MODEL_ID}")
    cfg = PreTrainedConfig.from_dict(
        {
            "type": "pi0",
            "pretrained_path": PI0_MODEL_ID,
            "device": str(device),
        }
    )

    policy = make_policy(cfg, ds_meta=ds_meta)
    policy.eval()

    # 3) Build pre- and post-processors for this dataset
    #    (new processors based on dataset stats and pi0 config)
    preproc, postproc = make_pre_post_processors(
        policy.config,
        dataset_stats=None,
    )

    # 4) Load a subset of the dataset (first episode only)
    print("Loading first episode frames...")
    dataset = LeRobotDataset(REPO_ID, episodes=[0])
    print(f"- Frames in first episode: {len(dataset)}")

    # 5) Offline inference loop
    task = "wash the pan in the sink"
    n_steps = min(10, len(dataset))  # keep it short for a quick demo

    print(f"\nRunning π₀ inference on {n_steps} frames with task: {task!r}\n")
    for i in range(n_steps):
        frame = dataset[i]

        batch = build_batch_from_frame(frame, task=task, index=i)

        # Preprocess inputs (normalization, tokenization, device move, etc.)
        policy_input = preproc(batch)

        with torch.no_grad():
            action = policy.select_action(policy_input)
            action = postproc(action)

        # `action` is a PolicyAction dict (e.g. {"action": tensor(...)}).
        # For a quick look, print shape and a small slice.
        action_tensor = action.get("action")
        if isinstance(action_tensor, torch.Tensor):
            # Move to CPU for printing
            action_np = action_tensor.detach().cpu().numpy()
            print(f"Step {i:02d}: action shape {action_np.shape}, first values {action_np.reshape(-1)[:5]}")
        else:
            print(f"Step {i:02d}: action (non-tensor) = {action}")


if __name__ == "__main__":
    main()
