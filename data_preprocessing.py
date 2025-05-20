import cv2
import numpy as np
import pandas as pd
import os

df=pd.read_csv('cleaned_imdb')
df = df.drop(columns=["Unnamed: 0"])

def preprocess_image(image_path, size=(224, 224)):
    print(f"Processing: {image_path}")  # Debugging output
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None
    resized_img = cv2.resize(img, size)
    normalized_img = resized_img / 255.0
    return normalized_img

base_path='/Users/daish/Downloads/imdb'
df['full_path']=df['full_path'].apply(lambda x: os.path.join(base_path,x))
print(df['full_path'].apply(os.path.exists).value_counts())
df=df[df['full_path'].apply(os.path.exists)]

df['processed_images']=df['full_path'].apply(preprocess_image)


df.to_pickle('processed_imdb.pkl')

