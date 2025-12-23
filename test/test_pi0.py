"""
Quick script to play with the π₀ (pi0) policy on a dataset,
without needing a real robot.

It loads:
- the NONHUMAN-RESEARCH/sarm_dataset_aloha_mobile_wash_pan dataset
- the lerobot/pi0_base policy
and runs offline inference on a few frames from the first episode.
Usage (from the repo root, after installing lerobot with `[pi]` extras):
"""

from typing import Any, Dict
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors
from adapt_dataset_for_pi0 import remap_observation_keys_for_pi0

REPO_ID = "NONHUMAN-RESEARCH/sarm_dataset_aloha_mobile_wash_pan"
PI0_MODEL_ID = "lerobot/pi0_base"


def select_device() -> torch.device:
    """Select a reasonable default device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_batch_from_frame(frame: Dict[str, Any],
                           task: str, index: int) -> Dict[str, Any]:
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

    # Recall that Pi Zero waits observation.images.base_0_rgb
    # observation.images.left_wrist_0_rgb
    # observation.images.right_wrist_0_rgb

    print(f"- Number of episodes: {len(ds_meta.episodes)}")

    # 2) Build π₀ config and policy (pretrained weights from the Hub)
    print(f"Loading π₀ policy from: {PI0_MODEL_ID}")

    policy = PI0Policy.from_pretrained(PI0_MODEL_ID)
    policy.config.device = str(device)
    policy.to(device)
    policy.eval()

    # 3) Build pre- and post-processors for this dataset
    #    (new processors based on dataset stats and pi0 config)
    preproc, postproc = make_pi0_pre_post_processors(
        config=policy.config,
        dataset_stats=None,
    )

    # 4) Load a subset of the dataset (first episode only)
    print("Loading first episode frames...")
    dataset = LeRobotDataset(REPO_ID, episodes=[0])
    print(f"- Frames in first episode: {len(dataset)}")

    # 5) Offline inference loop
    task = "Wash the pan in the sink"
    n_steps = min(10, len(dataset))  # keep it short for a quick demo

    print(f"\nRunning π₀ inference on {n_steps} frames with task: {task!r}\n")
    for i in range(n_steps):
        frame = dataset[i]

        # Adapt dataset keys to match π₀ expectations (in-memory only).
        frame = remap_observation_keys_for_pi0(frame)

        batch = build_batch_from_frame(frame, task=task, index=i)

        # Preprocess inputs (normalization, tokenization, device move, etc.)
        policy_input = preproc(batch)

        with torch.no_grad():
            action = policy.select_action(policy_input)
            action = postproc(action)

            policy_action = policy.select_action(policy_input)
            policy_action = postproc(policy_action)

        # `action` is a PolicyAction dict (e.g. {"action": tensor(...)}).
        # For a quick look, print shape and a small slice.

        if isinstance(policy_action, torch.Tensor):
            action_np = policy_action.detach().cpu().numpy()
            print(
                f"Step {i:02d}: action shape {action_np.shape}, "
                f"first values {action_np.reshape(-1)[:5]}"
            )
        else:
            print(f"Step {i:02d}: action (non-tensor) = {policy_action}")


if __name__ == "__main__":
    main()
