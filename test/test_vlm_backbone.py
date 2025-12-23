from transformers import AutoProcessor
from transformers.models.paligemma import PaliGemmaForConditionalGeneration

device = 'cuda'

model_id = "google/paligemma-3b-pt-224"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id).to(device)
processor = AutoProcessor.from_pretrained(model_id)

inputs = processor(text="I need help with math, can you help me?", images=None, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=64)
print(processor.tokenizer.decode(outputs[0], skip_special_tokens=True))
