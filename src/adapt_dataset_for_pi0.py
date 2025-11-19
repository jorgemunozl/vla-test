#!/usr/bin/env python
"""
Utility to adapt a dataset's observation keys to match π₀ (pi0) expectations.

Specifically tailored for:
  NONHUMAN-RESEARCH/sarm_dataset_aloha_mobile_wash_pan

This dataset uses camera names:
  - observation.images.cam_high
  - observation.images.cam_left_wrist
  - observation.images.cam_right_wrist

π₀ expects image features named:
  - observation.images.base_0_rgb
  - observation.images.left_wrist_0_rgb
  - observation.images.right_wrist_0_rgb

The helper below renames keys in-memory so you can feed frames to π₀
without modifying the dataset on disk.
"""

from __future__ import annotations

from typing import Any, Dict

try:
    # Optional imports for the small demo in __main__
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
except Exception:  # pragma: no cover - optional for pure key remapping usage
    LeRobotDataset = None  # type: ignore


# Mapping from dataset camera keys -> pi0 expected camera keys
CAMERA_RENAME_MAP: Dict[str, str] = {
    "observation.images.cam_high": "observation.images.base_0_rgb",
    "observation.images.cam_left_wrist": "observation.images.left_wrist_0_rgb",
    "observation.images.cam_right_wrist": "observation.images.right_wrist_0_rgb",
}


def remap_observation_keys_for_pi0(frame: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a copy of `frame` where known camera keys are duplicated under the
    names expected by π₀.

    The original keys are preserved; new keys are added only if they are not
    already present.
    """
    new_frame: Dict[str, Any] = dict(frame)
    for old_key, new_key in CAMERA_RENAME_MAP.items():
        if old_key in frame and new_key not in new_frame:
            new_frame[new_key] = frame[old_key]
    return new_frame


if __name__ == "__main__":
    # Small demo: show keys before/after remapping on the first frame.
    if LeRobotDataset is None:
        raise SystemExit(
            "lerobot is not installed in this environment. "
            "Install it first (e.g. `pip install -e '.[pi]'`) to run this demo."
        )

    REPO_ID = "NONHUMAN-RESEARCH/sarm_dataset_aloha_mobile_wash_pan"
    print(f"Loading dataset {REPO_ID} (first episode)...")
    ds = LeRobotDataset(REPO_ID, episodes=[0])

    frame0 = ds[0]
    print("\nOriginal frame keys (truncated):")
    print([k for k in frame0.keys() if k.startswith("observation.images.")])

    remapped = remap_observation_keys_for_pi0(frame0)
    print("\nRemapped frame keys (truncated):")
    print([k for k in remapped.keys() if k.startswith("observation.images.")])
