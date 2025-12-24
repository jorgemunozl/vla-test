import torch
from transformers import AutoTokenizer
from lerobot.policies.pi05.modeling_pi05 import PI05Policy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy = PI05Policy.from_pretrained("lerobot/pi05_base").to(device).eval()
paligemma = policy.model.paligemma_with_expert.paligemma.to(device).eval()


# Check that problematic wieght
with torch.no_grad():
    if hasattr(paligemma, "tie_weights"):
        paligemma.tie_weights()

    embed = paligemma.model.language_model.embed_tokens
    lm_head = paligemma.lm_head
    embed.weight = lm_head.weight

print("tied", embed.weight.data_ptr() == lm_head.weight.data_ptr())


# Tokenizer for Paligemma
tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

prompt = "Hello there"
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
