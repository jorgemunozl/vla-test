"""
PI05 VLM Subtask Training Script

Trains the PI05 Vision-Language Model on subtask prediction using
cross-entropy loss. No action training - purely VLM fine-tuning.

Usage:
    python train.py --image_dir ./images --epochs 10 --lr 1e-5
"""
import argparse
import logging
import os
import random
from glob import glob

import cv2
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from modeling_pi_05 import PI05Policy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# Task and Subtask Definitions
# ============================================================================

TASK = "Put the fruits in the basket"
SUBTASKS = [
    "Pick up the banana and put it in the basket",
    "Pick up the strawberry and put it in the basket",
    "Pick up the pear and put it in the basket",
    "Pick up the grape and put it in the basket",
]

# ============================================================================
# Dataset
# ============================================================================


class SubtaskDataset(Dataset):
    """
    Simple dataset that pairs images with subtasks for VLM training.

    Each sample contains:
    - An image loaded from disk
    - A tokenized sequence: "Task: {task}, Subtask: \n{subtask}"

    The newline character acts as separator - loss is computed only
    on tokens after the newline (the subtask portion).
    """

    def __init__(
        self,
        image_dir: str,
        task: str,
        subtasks: list[str],
        tokenizer,
        max_length: int = 128,
        image_key: str = "observation.images.top",
    ):
        """
        Args:
            image_dir: Directory containing images (*.png, *.jpg)
            task: High-level task description
            subtasks: List of possible subtasks
            tokenizer: HuggingFace tokenizer (PaliGemma)
            max_length: Maximum token sequence length
            image_key: Key for image in batch dict,
            the subtask is generated from the top image
        """
        self.task = task
        self.subtasks = subtasks
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_key = image_key

        # Find all images in directory
        image_patterns = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"]
        self.image_paths = []
        for pattern in image_patterns:
            self.image_paths.extend(glob(os.path.join(image_dir, pattern)))

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")

        logger.info(f"Found {len(self.image_paths)} images in {image_dir}")

        # Create samples: pair each image with each subtask
        # This creates len(images) * len(subtasks) samples
        self.samples = []
        for img_path in self.image_paths:
            for subtask in self.subtasks:
                self.samples.append((img_path, subtask))

        logger.info(f"Created {len(self.samples)} training samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, subtask = self.samples[idx]

        # Load and preprocess image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to tensor [H, W, C] -> [C, H, W] and normalize to [0, 1]
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # Create prompt with newline separator
        # Format: "Task: {task}, Subtask: \n{subtask}"
        prompt = f"Task: {self.task}, Subtask: \n{subtask}"

        # Tokenize
        tokenized = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        tokens = tokenized["input_ids"].squeeze(0)  # [seq_len]
        masks = tokenized["attention_mask"].squeeze(0)  # [seq_len]

        return {
            self.image_key: img,  # [C, H, W]
            "observation.language_tokens": tokens,  # [seq_len]
            "observation.language_attention_mask": masks,  # [seq_len]
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    # Stack all tensors
    collated = {}
    for key in batch[0].keys():
        collated[key] = torch.stack([item[key] for item in batch])
    return collated


# ============================================================================
# Training Loop
# ============================================================================


def train(
    image_dir: str,
    output_dir: str = "./checkpoints",
    model_id: str = "lerobot/pi05_base",
    num_epochs: int = 10,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.01,
    grad_clip_norm: float = 1.0,
    save_every: int = 1,
    log_every: int = 10,
    seed: int = 42,
    device: str | None = None,
):
    """
    Train PI05 VLM on subtask prediction.

    Args:
        image_dir: Directory containing training images
        output_dir: Directory to save checkpoints
        model_id: Pretrained model ID from HuggingFace
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        grad_clip_norm: Maximum gradient norm for clipping
        save_every: Save checkpoint every N epochs
        log_every: Log loss every N steps
        seed: Random seed
        device: Device to use (cuda/cpu)
    """
    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

    # Load pretrained policy
    logger.info(f"Loading pretrained model from {model_id}...")
    policy = PI05Policy.from_pretrained(model_id)
    policy.to(device)
    policy.train()

    # Create dataset and dataloader
    logger.info("Creating dataset...")
    dataset = SubtaskDataset(
        image_dir=image_dir,
        task=TASK,
        subtasks=SUBTASKS,
        tokenizer=tokenizer,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Keep simple for now
        collate_fn=collate_fn,
    )

    # Create optimizer
    optimizer = AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Training loop
    logger.info("Starting training...")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Total samples: {len(dataset)}")
    logger.info(f"  Steps per epoch: {len(dataloader)}")

    global_step = 0
    best_loss = float("inf")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            leave=True,
        )

        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass (subtask CE loss only)
            loss, loss_dict = policy.forward_subtask(batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), grad_clip_norm
                )

            # Optimizer step
            optimizer.step()

            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Log periodically
            if global_step % log_every == 0:
                avg_loss = epoch_loss / num_batches
                logger.info(
                    f"Step {global_step} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Avg Loss: {avg_loss:.4f}"
                )

        # End of epoch
        avg_epoch_loss = epoch_loss / num_batches
        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} completed | "
            f"Avg Loss: {avg_epoch_loss:.4f}"
        )

        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(
                output_dir, f"checkpoint_epoch_{epoch + 1}.pt"
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "model_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_epoch_loss,
                },
                checkpoint_path,
            )
            logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_path = os.path.join(output_dir, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "model_state_dict": policy.state_dict(),
                    "loss": best_loss,
                },
                best_path,
            )
            logger.info(f"New best model saved: {best_path} (loss: {best_loss:.4f})")

    logger.info("Training completed!")
    logger.info(f"Best loss: {best_loss:.4f}")

    return policy


# ============================================================================
# Main
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PI05 VLM on subtask prediction"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing training images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="lerobot/pi05_base",
        help="Pretrained model ID",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Gradient clipping norm",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=10,
        help="Log every N steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip,
        save_every=args.save_every,
        log_every=args.log_every,
        seed=args.seed,
        device=args.device,
    )
