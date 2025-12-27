import math
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# 1) Load PI05
policy = PI05Policy.from_pretrained("lerobot/pi05_base").to(DEVICE).eval()
paligemma = policy.model.paligemma_with_expert.paligemma.to(DEVICE).eval()

# (Optional but often needed due to tied weights conversions)
if hasattr(paligemma, "tie_weights"):
    paligemma.tie_weights()
try:
    paligemma.model.language_model.embed_tokens.weight = paligemma.lm_head.weight
except Exception:
    pass

# 2) Use the matching processor for the PaliGemma backbone
# (PI05 is based on a PaliGemma-style VLM; this processor gives the right image+text packing)
processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")

@torch.no_grad()
def score_text(image: Image.Image, prompt: str, completion: str) -> float:
    """
    Returns log P(completion | image, prompt) using teacher forcing.
    """
    # PaliGemma wants <image> in the text
    full = f"<image>\n{prompt}{completion}"
    inputs = processor(text=full, images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Labels = input_ids shifted for LM loss
    input_ids = inputs["input_ids"]
    labels = input_ids.clone()

    out = paligemma(**inputs, labels=labels)
    # out.loss is average CE over all tokens; convert to total logprob-ish score
    # (Higher is better, so we negate the loss*len)
    n_tokens = input_ids.numel()
    return float(-out.loss * n_tokens)

@torch.no_grad()
def pick_subtask(image_path: str, high_level_task: str, candidates: list[str]) -> tuple[str, float]:
    image = Image.open(image_path).convert("RGB")

    # Prompt format: make it “task -> next subtask:”
    prompt = f"Task: {high_level_task}\nNext subtask: "

    best = None
    best_score = -math.inf
    for c in candidates:
        s = score_text(image, prompt, c)
        if s > best_score:
            best_score = s
            best = c
    return best, best_score

if __name__ == "__main__":
    candidates = [
        "pick up the pillow",
        "adjust the blanket",
        "pick up the plate",
        "put the plate in the sink",
        "open the drawer",
    ]

    subtask, score = pick_subtask(
        image_path="scene.jpg",
        high_level_task="clean the bedroom",
        candidates=candidates,
    )
    print("predicted subtask:", subtask)
    print("score:", score)
