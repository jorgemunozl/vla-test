from xhuman.policies.pi05.processor_pi05 import make_pi05_pre_post_processors_ki
import torch
from transformers import AutoTokenizer
from xhuman.policies.pi05.configuration_pi05 import PI05Config

config = PI05Config()

task = "Move to the door."
dummy_batch = {
    "frame_index": 13,
    "task": task,
    "observation.images.top": torch.randn(3, 224, 224),
    "observation.state": torch.randn(14),
}

preprocessor, postprocessor = make_pi05_pre_post_processors_ki(
    config,
)

batch = preprocessor(dummy_batch)
prompt = batch["observation.language.tokens"]

tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
prompt = tokenizer.batch_decode(prompt, skip_special_tokens=True)
print(f"Prompt: {prompt}")