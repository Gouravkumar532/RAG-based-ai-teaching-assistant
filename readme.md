# How to use this rag ai teaching assistant

## Step 1 - Collect your videos
Move all your videos files to the videos folder

## Step 2 - Convert to mp3
Convert all the video files to mp3 by running video_to_mp

## Step 3 - Convert mp3 to json
Convert all the mp3 files to json by running mp3_to_json

## Step 4 - Convert the json files to Vectors
Use the file preprocess_json to convert the json files to a datafram with embeddings and save it as a joblib pickle

## Step 5 - Prompt generation and feeding to LLM
Read the joblib file anad load it  into the memory. Then create a revelant prompt as per the use query and feed it to the LLM