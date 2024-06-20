from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pandas as pd
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError
from google.cloud import firestore
from google.cloud import storage
import os
from dotenv import load_dotenv
import uvicorn

load_dotenv()

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'serviceAccountFireStore.json'

# Inisialisasi klien Firestore
db = firestore.Client()
app = FastAPI()

# Mendaftarkan fungsi mse sebagai objek khusus
def get_mse():
    return MeanSquaredError(name='mse')

def take_model(bucket_name, model_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(model_name)
    local_model_filename = 'new_model.h5'
    blob.download_to_filename(local_model_filename)
    return local_model_filename

model_path = take_model('sellucose-model-ml', 'new_model.h5')
model = load_model(model_path, custom_objects={'mse': get_mse})

def load_and_preprocess_data():
    books_ref = db.collection('books').limit(50000)
    books = [doc.to_dict() for doc in books_ref.stream()]
    books_df = pd.DataFrame(books)

    # Ambil data rating dari Firestore
    ratings_ref = db.collection('rated_books')
    ratings = [doc.to_dict() for doc in ratings_ref.stream()]
    ratings_df = pd.DataFrame(ratings)

    books_data = books_df.merge(ratings_df, on="ISBN")
    df = books_data.copy()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.drop(columns=["ISBN", "Year-Of-Publication", "Image-URL-S", "Image-URL-M"], axis=1, inplace=True)
    df.drop(index=df[df["Book-Rating"] == 0].index, inplace=True)
    df["Book-Title"] = df["Book-Title"].apply(lambda x: re.sub("[\W_]+", " ", x).strip())

    user_ids = df['User-ID'].unique().tolist()
    user_id_to_index = {user_id: index for index, user_id in enumerate(user_ids)}

    book_titles = df['Book-Title'].unique().tolist()
    book_title_to_index = {book_title: index for index, book_title in enumerate(book_titles)}
    index_to_book_title = {index: book_title for index, book_title in enumerate(book_titles)}

    df['user_index'] = df['User-ID'].map(user_id_to_index)
    df['book_index'] = df['Book-Title'].map(book_title_to_index)

    return df, user_id_to_index, book_title_to_index, index_to_book_title

# Load and preprocess the dataset
df, user_id_to_index, book_title_to_index, index_to_book_title = load_and_preprocess_data()

# Define the recommendation function
def recommend_books(user_id, model, user_id_to_index, index_to_book_title, top_n=5):
    # Get the index of the user
    user_idx = user_id_to_index.get(user_id, None)
    if user_idx is None:
        return []

    # Create input arrays for the model
    user_array = np.array([user_idx] * len(book_title_to_index))
    book_array = np.arange(len(book_title_to_index))

    # Predict ratings for all books
    predicted_ratings = model.predict([user_array, book_array])

    # Get the top N book recommendations
    top_indices = predicted_ratings.flatten().argsort()[-top_n:][::-1]
    top_books = [index_to_book_title[idx] for idx in top_indices]

    return top_books

@app.get("/predict")
def predict(user_id: int):
    # Validate if the user ID exists in the dataset
    if user_id not in user_id_to_index:
        raise HTTPException(status_code=400, detail="User ID not found in the dataset")

    # Get recommendations
    recommended_books = recommend_books(user_id, model, user_id_to_index, index_to_book_title, top_n=5)

    return {"top_books": recommended_books}

# Run the FastAPI application using uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
