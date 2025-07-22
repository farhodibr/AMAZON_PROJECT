import pandas as pd

#load the full JSONL
df = pd.read_json(r"C:\CUNY_MSDS\DATA612\VIDEO_GAMES_PROJECT\data\meta_Amazon_Fashion.jsonl", lines=True)

# extract only the columns needed
df_small = pd.DataFrame({
    "parent_asin": df["parent_asin"],
    "title": df["title"],
    "image_url": df["images"].apply(lambda imgs: imgs[0]["large"] if isinstance(imgs, list) and imgs else None)
})

# drop rows with missing URLs
df_small = df_small.dropna(subset=["image_url"])

# save as Parquet or JSON (much smaller than original 1.4 GB)
df_small.to_parquet("meta_small.parquet", index=False)
