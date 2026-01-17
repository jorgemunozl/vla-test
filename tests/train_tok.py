def load_actions_from_dataset(
    repo_id: str,
    batch_size: int = 8192,
    action_horizon: int = 50,
    action_dim: int = 14,
) -> np.ndarray:
    """
    Load actions from the dataset and return them in the specified shape.

    Args:
        repo_id: HuggingFace dataset repository ID
        batch_size: Number of action sequences to return (B)
        action_horizon: Length of each action sequence (action_horizon)
        action_dim: Dimension of each action vector (action_dim)

    Returns:
        numpy array of shape (B, action_horizon, action_dim) containing actions
    """
    # Load the dataset
    dataset = XHumanDataset(repo_id=repo_id)
    dataset._ensure_hf_dataset_loaded()

    # Extract actions from the dataset
    actions = dataset.hf_dataset["action"]

    # Convert to numpy if needed and stack actions
    action_list = []
    for action in actions:
        if isinstance(action, torch.Tensor):
            action_np = action.cpu().numpy()
        else:
            action_np = np.array(action)
        action_list.append(action_np)

    # Stack all actions into a single array
    all_actions = np.stack(action_list)  # Shape: (num_frames, action_dim)

    # Reshape into sequences of action_horizon length
    num_sequences = len(all_actions) // action_horizon
    if num_sequences == 0:
        raise ValueError(
            f"Dataset has {len(all_actions)} frames, but need at least "
            f"{action_horizon} frames for one sequence"
        )

    # Truncate to fit exact sequences
    total_frames_needed = num_sequences * action_horizon
    all_actions = all_actions[:total_frames_needed]

    # Reshape to (num_sequences, action_horizon, action_dim)
    sequences = all_actions.reshape(num_sequences, action_horizon, action_dim)

    # Sample or pad to batch_size
    if num_sequences >= batch_size:
        # Randomly sample batch_size sequences
        indices = np.random.choice(
            num_sequences, size=batch_size, replace=False
        )
        result = sequences[indices]
    else:
        # Repeat sequences to reach batch_size
        num_repeats = (batch_size // num_sequences) + 1
        repeated = np.tile(sequences, (num_repeats, 1, 1))
        result = repeated[:batch_size]

    print(result.shape)

    return result


def train_fast_tokenizer_processor_step():
    """
    Constructs pre-processor and
    post-processor pipelines for the PI0 policy.
    """
    # data: (B, action_horizon, action_dim)
    repo_id = "NONHUMAN-RESEARCH/pick-and-place-fruits_v2_cleaned"
    data = load_actions_from_dataset(repo_id=repo_id)
    # Obtain a huge batch of continous actions
    tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast")
    tokenizer.fit(data)
    tokenizer.save_pretrained("/home/jorge/project/XHUMAN")
    tokenizer.push_to_hub("jorgemunozl/fast_tokenizer")


if __name__ == "__main__":
    train_fast_tokenizer_processor_step()