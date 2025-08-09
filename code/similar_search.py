import numpy as np
import faiss
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# === Paths ===
INDEX_PATH = "E:/projects/Visual Search Engine/embeddings/faiss_index.index"

FILENAME_PATH = "E:/projects/Visual Search Engine/embeddings/image_filenames.npy"
QUERY_IMAGE_PATH = "E:/projects/Visual Search Engine/Data/Images_Train/img_26_label_Dress.png"  # your input image

# === Load FAISS index & filenames ===
index = faiss.read_index(INDEX_PATH)
filenames = np.load(FILENAME_PATH)

# === Load CLIP model and processor ===
device = "cpu"  # or "cuda" if your GPU can handle it
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",use_safetensors=True).to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32",use_fast=True)

# === Load and process query image ===
image = Image.open(QUERY_IMAGE_PATH)
inputs = processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    query_embedding = model.get_image_features(**inputs)

# === Normalize and convert to float32 ===
query_embedding = query_embedding / query_embedding.norm(p=2, dim=-1, keepdim=True)
query_np = query_embedding.cpu().numpy().astype("float32")

print("Query dtype:", query_np.dtype)         # must be float32
print("Query shape:", query_np.shape)         # must be (1, 512)
print("Index dimension:", index.d)            # must be 512
print("Index size:", index.ntotal)            # >0

assert query_np.dtype == np.float32, "Query must be float32"
assert query_np.shape == (1, index.d), f"Query shape mismatch: {query_np.shape} vs index dim {index.d}"

k = 5
distances, indices = index.search(query_np, k)

# === Output ===
print("\n🔍 Top matches:")
for rank, idx in enumerate(indices[0]):
    print(f"{rank+1}. {filenames[idx]}  (distance: {distances[0][rank]:.4f})")