import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    data = pd.merge(movies, ratings, on="movieId")
    pivot = data.pivot_table(index='userId', columns='title', values='rating').fillna(0)
    similarity = cosine_similarity(pivot.T)
    sim_df = pd.DataFrame(similarity, index=pivot.columns, columns=pivot.columns)
    return sim_df

similarity_df = load_data()

st.title("🎬 Movie Recommender App")

movie_name = st.selectbox("Choose a movie you like:", similarity_df.columns.sort_values())

if st.button("Recommend 🎥"):
    st.subheader(f"Because you liked **{movie_name}**, you might also enjoy:")
    similar_movies = similarity_df[movie_name].sort_values(ascending=False)[1:6]
    for i, (movie, score) in enumerate(similar_movies.items(), start=1):
        st.write(f"{i}. **{movie}**  (Similarity: {score:.2f})")
