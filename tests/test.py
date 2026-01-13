"""
Simple test for PI05Model.forward method.
"""
import torch
import numpy as np
from unittest.mock import Mock, patch

from xhuman.policies.pi05.modeling_pi_05 import PI05Model
from xhuman.policies.pi05.configuration_pi05 import PI05Config


def test_pi05_model_forward():
    """Test PI05Model.forward with synthetic inputs."""
    # Create minimal config
    config = PI05Config(
        device="cuda",  # Use CUDA for testing
        chunk_size=10,  # Small chunk size for faster testing
        n_action_steps=10,  # Must be <= chunk_size
        max_action_dim=8,
        dtype="float32",
        compile_model=False,  # Disable compilation for testing
        gradient_checkpointing=False,
    )
    config.validate_features()

    batch_size = 2  # Define batch size for test inputs
    discrete_action_seq_len = 5  # Length of discrete action sequence
    

    def mock_fast_tokenizer(actions):
        """
        Mock fast_tokenizer that returns discrete action tokens.
        
        Args:
            actions: torch.Tensor of shape (B, chunk_size, action_dim)
                    Continuous action values
            
        Returns:
            numpy.ndarray of shape (B, discrete_action_seq_len)
            Discrete token IDs representing the actions
        """
        # Extract batch_size from actions shape: (B, chunk_size, action_dim)
        batch_size = actions.shape[0]
        # Return numpy array of discrete tokens (random for testing)
        return np.random.randint(100, 200, (batch_size, discrete_action_seq_len))

    
    mock_processor = Mock()
    mock_processor.side_effect = mock_fast_tokenizer

    with patch('xhuman.policies.pi05.modeling_pi_05.AutoProcessor.from_pretrained', return_value=mock_processor):
        # Create model - inside this context, AutoProcessor.from_pretrained returns mock_processor
        model = PI05Model(config, rtc_processor=None)
        model.train()  # Set to training mode

        # Replace the fast_tokenizer attribute with our mock function directly
        # This ensures model.fast_tokenizer(actions) calls our mock function
        model.fast_tokenizer = mock_fast_tokenizer

        # Create synthetic inputs
        seq_len = 20
        chunk_size = config.chunk_size
        action_dim = config.max_action_dim

        # Images: list of tensors, each (B, C, H, W)
        images = [
            torch.randn(batch_size, 3, 224, 224, dtype=torch.float32),
        ]
        img_masks = [
            torch.ones(batch_size, dtype=torch.bool),
        ]

        # Tokens: (B, seq_len) - prompt tokens with padding
        # Include some padding (0s) where discrete actions will be inserted
        tokens = torch.randint(10, 1000, (batch_size, seq_len), dtype=torch.long)
        # Set some tokens to 0 to mark padding positions
        tokens[:, -5:] = 0

        # Masks: (B, seq_len) - attention masks
        masks = torch.ones(batch_size, seq_len, dtype=torch.bool)
        masks[:, -5:] = False  # Mark padding positions

        # Actions: (B, chunk_size, action_dim)
        actions = torch.randn(batch_size, chunk_size, action_dim, dtype=torch.float32) * 0.1

        # Call forward
        ce_loss, flow_matching_loss = model.forward(
            images=images,
            img_masks=img_masks,
            tokens=tokens,
            masks=masks,
            actions=actions,
        )

        # Check outputs
        assert isinstance(ce_loss, torch.Tensor), "ce_loss should be a tensor"
        assert isinstance(flow_matching_loss, torch.Tensor), "flow_matching_loss should be a tensor"
        assert ce_loss.shape == (batch_size,), f"ce_loss shape should be ({batch_size},), got {ce_loss.shape}"
        assert flow_matching_loss.shape == (batch_size,), f"flow_matching_loss shape should be ({batch_size},), got {flow_matching_loss.shape}"
        assert ce_loss.dtype == torch.float32, f"ce_loss dtype should be float32, got {ce_loss.dtype}"
        assert flow_matching_loss.dtype == torch.float32, f"flow_matching_loss dtype should be float32, got {flow_matching_loss.dtype}"

        # Check that losses are non-negative
        assert (ce_loss >= 0).all(), "ce_loss should be non-negative"
        assert (flow_matching_loss >= 0).all(), "flow_matching_loss should be non-negative"

        print("✓ Test passed!")
        print(f"  ce_loss shape: {ce_loss.shape}, mean: {ce_loss.mean().item():.4f}")
        print(f"  flow_matching_loss shape: {flow_matching_loss.shape}, mean: {flow_matching_loss.mean().item():.4f}")


if __name__ == "__main__":
    test_pi05_model_forward()
