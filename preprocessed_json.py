import requests
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def create_embedding(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )

    r.raise_for_status()

    return r.json()["embeddings"]


jsons = os.listdir("newjsons")

my_dicts = []
chunk_id = 0

BATCH_SIZE = 200

for json_file in jsons:

    with open(f"newjsons/{json_file}", encoding="utf-8") as f:
        content = json.load(f)

    chunks = content["chunks"]
    texts = [c["text"] for c in chunks]

    for start in range(0, len(texts), BATCH_SIZE):

        batch = texts[start:start + BATCH_SIZE]

        print(
            f"{json_file}: "
            f"{start + 1}-{start + len(batch)} / {len(texts)}"
        )

        embeddings = create_embedding(batch)

        for i, embedding in enumerate(embeddings):

            index = start + i

            chunks[index]["chunk_id"] = chunk_id
            chunks[index]["embedding"] = embedding

            my_dicts.append(chunks[index])

            chunk_id += 1
            
    print(f"Finished {json_file}")

print(f"\nTotal chunks: {len(my_dicts)}")

df = pd.DataFrame.from_records(my_dicts)
#print(df)
#saving the dataframe
joblib.dump(df, 'embeddings.joblib')
