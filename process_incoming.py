import requests
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


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

def inference(prompt):

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content

df = joblib.load('embeddings.joblib')

incoming_query = input("Ask a question : ")
question_embedding = create_embedding([incoming_query])[0]

#Find similarites of question embeddings with other embeddings

similarities = cosine_similarity(np.vstack(df['embedding'].values),[question_embedding]).flatten()

top_Results = 5
max_indx = similarities.argsort()[::-1][0:top_Results]

#print(max_indx)

new_df = df.loc[max_indx]
#print(new_df[["title", "number", "text"]])

prompt = f'''I am teaching werb development using Sigma web development course.
Here are video subtitle chunks containing video title, video number,start time in seconds,
end time in seconds, the text at that time : 

{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
-------------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention
 the above format, its just for you) where and how much 
content is taught in which video ( in which video and at what timestamp ) and guide the user to go to that
particular video. If user asks unrelated question, tell him that you can only answer
questions related to the course
'''

# for index,item in new_df.iterrows():
#     print(index,item["title"],item["number"],item["text"],item["start"],item["end"])

with open("prompt.txt","w") as f:
    f.write(prompt)

response = inference(prompt)

print(response)

with open("response.txt", "w", encoding="utf-8") as f:
    f.write(response)