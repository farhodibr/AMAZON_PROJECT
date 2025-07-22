import os
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, Repository
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import requests
from io import BytesIO
from tqdm import tqdm

# parameters
DATA_SUBSET_SIZE = 1000
IMAGE_DIR = "product_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# download metadata
from huggingface_hub import hf_hub_download
meta_path = hf_hub_download(
    repo_id="glavvrach79/my-recsys-data",
    repo_type="dataset",
    filename="meta_Amazon_Fashion.jsonl",
    subfolder="data"
)
meta_df = pd.read_json(meta_path, lines=True)
meta_df['image_url'] = meta_df['images'].apply(
    lambda x: x[0]['large'] if x and isinstance(x, list) and x[0].get('large') else None
)
meta_df = meta_df.dropna(subset=['image_url']).head(DATA_SUBSET_SIZE)

# download images
image_embeddings = {}
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]).eval()
preprocess = weights.transforms()

for _, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
    asin = row['parent_asin']
    url  = row['image_url']
    try:
        resp = requests.get(url, timeout=5)
        img  = Image.open(BytesIO(resp.content)).convert("RGB")
        t   = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            feat = feature_extractor(t).squeeze().numpy()
        image_embeddings[asin] = feat
    except Exception:
        continue

# build arrays and save
item_ids = list(image_embeddings.keys())
emb_matrix = np.stack([image_embeddings[a] for a in item_ids])

# save separate files
np.save("image_embeddings.npy", emb_matrix)
pd.DataFrame({"item_id": item_ids}).to_csv("item_ids.csv", index=False)

#single compressed archive
np.savez_compressed("image_index.npz",
                    embeddings=emb_matrix,
                    ids=np.array(item_ids, dtype=object))

print(f"Saved {emb_matrix.shape[0]} embeddings of dimension {emb_matrix.shape[1]}")
