"""
Script to extract and display information from a LeRobot dataset.

This script loads a lerobot dataset and extracts comprehensive
information including:
- Dataset metadata (info, tasks, episodes)
- Feature keys and shapes
- Statistics for normalization
- Episode information
- Sample data inspection
"""

import json
from pathlib import Path
from typing import Any

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pprint import pprint


def print_section(title: str, width: int = 80):
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def extract_dataset_info(dataset: LeRobotDataset) -> dict[str, Any]:
    """
    Extract comprehensive information from a LeRobot dataset.

    Args:
        dataset: The LeRobotDataset instance

    Returns:
        Dictionary containing all extracted information
    """
    info = {}

    # Basic dataset properties
    info["dataset_length"] = len(dataset)
    repo_id = "N/A"
    if hasattr(dataset, "repo_id"):
        repo_id = dataset.repo_id
    info["repo_id"] = repo_id

    # Metadata information
    if hasattr(dataset, "meta"):
        meta = dataset.meta

        # Dataset info
        if hasattr(meta, "info") and meta.info:
            info["dataset_info"] = dict(meta.info)

        # Tasks
        if hasattr(meta, "tasks") and meta.tasks:
            info["tasks"] = meta.tasks

        # Episodes
        if hasattr(meta, "episodes") and meta.episodes:
            episodes = meta.episodes
            info["num_episodes"] = len(episodes)
            info["episode_lengths"] = [
                ep.get("dataset_to_index", 0) - ep.get("dataset_from_index", 0)
                for ep in episodes.values()
            ]
            info["total_frames"] = sum(info["episode_lengths"])
            info["episode_info"] = {
                idx: {
                    "from_index": ep.get("dataset_from_index"),
                    "to_index": ep.get("dataset_to_index"),
                    "length": (
                        ep.get("dataset_to_index", 0)
                        - ep.get("dataset_from_index", 0)
                    ),
                }
                for idx, ep in episodes.items()
            }

        # Statistics
        if hasattr(meta, "stats") and meta.stats:
            info["statistics"] = {}
            for key, stats_dict in meta.stats.items():
                info["statistics"][key] = {
                    k: (
                        v.tolist() if isinstance(v, torch.Tensor) else v
                    )
                    for k, v in stats_dict.items()
                }

        # Camera and video keys
        if hasattr(meta, "camera_keys"):
            info["camera_keys"] = list(meta.camera_keys)
        if hasattr(meta, "video_keys"):
            info["video_keys"] = list(meta.video_keys)

    # Feature keys and shapes from hf_dataset
    if hasattr(dataset, "hf_dataset"):
        hf_dataset = dataset.hf_dataset
        info["feature_keys"] = list(hf_dataset.features.keys())

        # Get shapes from first sample
        if len(hf_dataset) > 0:
            sample = hf_dataset[0]
            info["feature_shapes"] = {}
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    info["feature_shapes"][key] = list(value.shape)
                elif isinstance(value, (list, tuple)) and len(value) > 0:
                    if isinstance(value[0], torch.Tensor):
                        # First 3 items
                        info["feature_shapes"][key] = [
                            list(v.shape) for v in value[:3]
                        ]
                    else:
                        type_name = type(value[0]).__name__
                        info["feature_shapes"][key] = (
                            f"list of {type_name}, length={len(value)}"
                        )
                else:
                    info["feature_shapes"][key] = type(value).__name__

    # Check if episodes filter is applied
    if hasattr(dataset, "episodes") and dataset.episodes is not None:
        info["filtered_episodes"] = dataset.episodes

    return info


def print_dataset_info(info: dict[str, Any], verbose: bool = False):
    """Print extracted dataset information in a formatted way."""

    print_section("Dataset Overview")
    print(f"Repository ID: {info.get('repo_id', 'N/A')}")
    print(f"Dataset Length: {info.get('dataset_length', 'N/A'):,}")

    if "num_episodes" in info:
        print(f"Number of Episodes: {info['num_episodes']}")
        print(f"Total Frames: {info.get('total_frames', 'N/A'):,}")
        if info["episode_lengths"]:
            min_len = min(info['episode_lengths'])
            max_len = max(info['episode_lengths'])
            print(f"Episode Length Range: {min_len} - {max_len} frames")
            avg_len = (
                sum(info['episode_lengths']) / len(info['episode_lengths'])
            )
            print(f"Average Episode Length: {avg_len:.1f} frames")

    if "filtered_episodes" in info:
        print(f"Filtered Episodes: {info['filtered_episodes']}")

    if "dataset_info" in info:
        print_section("Dataset Info")
        pprint(info["dataset_info"], width=80)

    if "tasks" in info:
        print_section("Tasks")
        if isinstance(info["tasks"], dict):
            for task_id, task_data in info["tasks"].items():
                print(f"Task {task_id}: {task_data}")
        else:
            print(info["tasks"])

    if "episode_info" in info and verbose:
        print_section("Episode Details")
        # Show first 10
        for ep_idx, ep_info in list(info["episode_info"].items())[:10]:
            print(f"Episode {ep_idx}: {ep_info}")
        if len(info["episode_info"]) > 10:
            remaining = len(info['episode_info']) - 10
            print(f"... and {remaining} more episodes")

    if "feature_keys" in info:
        print_section("Feature Keys")
        print(f"Total Features: {len(info['feature_keys'])}")
        for key in info["feature_keys"]:
            print(f"  - {key}")

    if "feature_shapes" in info:
        print_section("Feature Shapes (from first sample)")
        for key, shape in info["feature_shapes"].items():
            print(f"  {key}: {shape}")

    if "camera_keys" in info:
        print_section("Camera Keys")
        for key in info["camera_keys"]:
            print(f"  - {key}")

    if "video_keys" in info:
        print_section("Video Keys")
        for key in info["video_keys"]:
            print(f"  - {key}")

    if "statistics" in info:
        print_section("Statistics (for normalization)")
        for key, stats in info["statistics"].items():
            print(f"\n  {key}:")
            for stat_name, stat_value in stats.items():
                if isinstance(stat_value, list):
                    if len(stat_value) > 10:
                        print(
                            f"    {stat_name}: {stat_value[:5]} ... "
                            f"(length={len(stat_value)})"
                        )
                    else:
                        print(f"    {stat_name}: {stat_value}")
                else:
                    print(f"    {stat_name}: {stat_value}")


def inspect_sample(dataset: LeRobotDataset, index: int = 0):
    """Inspect a specific sample from the dataset."""
    print_section(f"Sample Inspection (index={index})")

    if index >= len(dataset):
        msg = (
            f"Error: Index {index} is out of range "
            f"(dataset length: {len(dataset)})"
        )
        print(msg)
        return

    try:
        sample = dataset[index]
        print(f"Sample keys: {list(sample.keys())}")
        print()

        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}:")
                print(f"  Shape: {value.shape}")
                print(f"  Dtype: {value.dtype}")
                if value.numel() < 20:  # Only print small tensors
                    print(f"  Values: {value}")
                else:
                    min_val = value.min().item()
                    max_val = value.max().item()
                    mean_val = value.mean().item()
                    print(
                        f"  Min: {min_val:.4f}, Max: {max_val:.4f}, "
                        f"Mean: {mean_val:.4f}"
                    )
            elif isinstance(value, (list, tuple)):
                type_name = type(value).__name__
                print(f"{key}: {type_name} of length {len(value)}")
                if len(value) > 0 and isinstance(value[0], torch.Tensor):
                    print(f"  First element shape: {value[0].shape}")
            else:
                print(f"{key}: {type(value).__name__} = {value}")
            print()
    except Exception as e:
        print(f"Error inspecting sample: {e}")
        import traceback
        traceback.print_exc()


def detect_repo_id_from_path(
    local_path: str | Path
) -> tuple[str | None, str | None]:
    """
    Try to detect repo_id and root from local path structure.
    Looks for patterns like: .../lerobot/user/dataset-name/

    Returns:
        (repo_id, root) tuple or (None, None) if detection fails
    """
    local_path = Path(local_path).resolve()
    parts = local_path.parts

    # Look for 'lerobot' in path
    if 'lerobot' in parts:
        idx = parts.index('lerobot')
        # Root should be the directory containing 'lerobot'
        root = Path(*parts[:idx + 1])

        # Get repo_id from parts after 'lerobot'
        if idx + 1 < len(parts):
            repo_id = parts[idx + 1]
            # Check if it's user/dataset-name format (two parts)
            if idx + 2 < len(parts):
                repo_id = f"{parts[idx + 1]}/{parts[idx + 2]}"
            return repo_id, str(root)

    # If path looks like a dataset directory (has meta/ or data/)
    if (local_path / "meta").exists() or (local_path / "data").exists():
        # Check if parent is 'lerobot'
        parent = local_path.parent
        if parent.name == "lerobot":
            # repo_id is the dataset directory name
            repo_id = local_path.name
            root = str(parent.parent)
            return repo_id, root

    return None, None


def main():
    # Configuration - modify these variables as needed
    # Option 1: Use HuggingFace repository ID
    # (dataset will be loaded from cache)
    repo_id = "user/dataset-name"  # HuggingFace repository ID

    # Option 2: Use local dataset path (set this if dataset is local)
    # If set, repo_id will be auto-detected from path if possible
    # e.g., "/path/to/.cache/huggingface/lerobot/user/dataset-name"
    local_path = "/home/jorge/NONHUMAN/ds/test/snapshots/cfc9dc0860527358a018d660506bfa743b7a1c5a/"

    # Option 3: Specify custom root directory for cache
    # (default: ~/.cache/huggingface/lerobot)
    root = None

    # Filter specific episodes (e.g., [0, 1, 2]) or None for all
    episodes = None
    revision = None  # Dataset revision/branch to load
    verbose = False  # Show detailed episode information
    # Inspect a specific sample at this index (or None)
    inspect_sample_idx = None
    # Save extracted information to JSON file (or None)
    save_json_path = None

    # Handle local path if provided
    if local_path is not None:
        local_path = Path(local_path)
        if not local_path.exists():
            print(f"Error: Local path does not exist: {local_path}")
            return

        # Try to detect repo_id and root from path
        detected_repo_id, detected_root = detect_repo_id_from_path(local_path)
        if detected_repo_id and detected_root:
            repo_id = detected_repo_id
            root = detected_root
            print(f"Detected repo_id: {repo_id} from local path")
            print(f"Using root: {root}")
        else:
            # Fallback: use directory name as repo_id
            if repo_id == "user/dataset-name":
                repo_id = local_path.name
            # Try to find lerobot parent directory
            current = local_path
            while current.parent != current:
                if current.name == "lerobot":
                    root = str(current.parent)
                    break
                current = current.parent
            else:
                # If no lerobot found, use parent of dataset dir as root
                root = str(local_path.parent)
            print(f"Using repo_id: {repo_id}, root: {root}")

    # Load dataset
    print(f"Loading dataset: {repo_id}")
    if root:
        print(f"Using root directory: {root}")
    try:
        dataset = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            revision=revision,
        )
        print("Dataset loaded successfully!")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nTips:")
        print("1. For HF datasets: set repo_id and leave root=None")
        print("2. For local datasets: set local_path to the dataset directory")
        print("   (e.g., ~/.cache/huggingface/lerobot/user/dataset-name)")
        print("3. Or set root to the parent of the dataset directory")
        import traceback
        traceback.print_exc()
        return

    # Extract information
    info = extract_dataset_info(dataset)

    # Print information
    print_dataset_info(info, verbose=verbose)

    # Inspect sample if requested
    if inspect_sample_idx is not None:
        inspect_sample(dataset, inspect_sample_idx)

    # Save to JSON if requested
    if save_json_path:
        # Convert torch tensors to lists for JSON serialization
        json_info = json.loads(json.dumps(info, default=str))
        with open(save_json_path, "w") as f:
            json.dump(json_info, f, indent=2, default=str)
        print(f"\nInformation saved to: {save_json_path}")


if __name__ == "__main__":
    main()
