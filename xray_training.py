from transformers import BlipProcessor, BlipForConditionalGeneration
from datasets import load_dataset
import torch

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
dataset = load_dataset("eltorio/ROCOv2-radiology", split="train[:500]")
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
model.train()
for epoch in range(1):
    for example in dataset:
        image = example["image"]
        caption = example["caption"]
        inputs = processor(images=image, text=caption, return_tensors="pt", padding=True)
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch} loss: {loss.item()}")
model.save_pretrained("./blip-xray-finetuned")