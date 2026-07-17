from datasets import load_dataset
dataset = load_dataset("eltorio/ROCOv2-radiology")
print(dataset[0])
dataset = load_dataset("alkzar90/NIH-Chest-X-ray-dataset", split="train")
print(dataset[0])