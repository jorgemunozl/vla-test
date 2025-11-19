import torch
from transformers import AutoTokenizer
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

# Load your policy as usual
policy = PI05Policy.from_pretrained("lerobot/pi05_base")
device = policy.config.device
paligemma = policy.model.paligemma_with_expert.paligemma.to(device).eval()

# Use the same tokenizer as PI05
tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

prompt = "Hello there!"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    generated_ids = paligemma.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=True,
        temperature=0.7,
    )

text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(text)
