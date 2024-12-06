
import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import zipfile

st.title("IMDb Top Movies Recommendation System")

# File paths
movies_file = 'movies.zip'
ratings_file = 'ratings.zip'

# Read specific file from the zip archive
with zipfile.ZipFile(movies_file, 'r') as z:
    with z.open('movies.csv') as f:
        movies = pd.read_csv(f)

with zipfile.ZipFile(ratings_file, 'r') as z:
    with z.open('ratings.csv') as f:
        ratings = pd.read_csv(f)


# Preprocess movies data
movies['title'] = movies['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()

# Aggregate ratings
movie_ratings = ratings.groupby('movieId').agg({'rating': 'mean', 'userId': 'count'}).reset_index()
movie_ratings.columns = ['movieId', 'avg_rating', 'rating_count']

# Merge movies with ratings
movies = movies.merge(movie_ratings, on='movieId', how='left')
movies['avg_rating'] = movies['avg_rating'].fillna(0)
movies['rating_count'] = movies['rating_count'].fillna(0)

# Build recommendation system
vectorizer = TfidfVectorizer(token_pattern='[a-zA-Z0-9]+')
genre_matrix = vectorizer.fit_transform(movies['genres'])
knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(genre_matrix)

# User input and recommendations
movie_name = st.text_input("Enter a movie you have watched:")

if st.button("Recommend Similar Movies"):
    if movie_name in movies['title'].values:
        movie_idx = movies[movies['title'] == movie_name].index[0]
        _, indices = knn.kneighbors(genre_matrix[movie_idx])

        st.write("4 recommended similar movies:")
        for idx in indices[0][1:]: 
            st.write(movies.iloc[idx]['title'])
    else:
        st.write("Movie not found. Please try another one.")


