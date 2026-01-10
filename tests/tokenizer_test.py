from transformers import AutoProcessor
import numpy as np

tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast",
trust_remote_code=True)


dummy_actions = np.random.randn(3, 10, 2)

print("dummy_actions shape:", dummy_actions.shape)
print("dummy_actions:", dummy_actions)
tokenized_actions = tokenizer(dummy_actions)
print("tokenized_actions:", tokenized_actions)

print("tokenized_actions:", tokenized_actions)

print("decoded_actions:", tokenizer.decode(tokenized_actions))
