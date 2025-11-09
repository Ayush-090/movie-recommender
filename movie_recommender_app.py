import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    # Load both CSV files
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")

    # Clean up and normalize column names
    movies.columns = movies.columns.str.strip()
    ratings.columns = ratings.columns.str.strip()

    # Rename possible variants of movieId
    rename_map = {'movie_id': 'movieId', 'MovieID': 'movieId', 'id': 'movieId'}
    movies.rename(columns=rename_map, inplace=True)
    ratings.rename(columns=rename_map, inplace=True)

    # Ensure required columns exist
    if "movieId" not in movies.columns or "movieId" not in ratings.columns:
        st.error(f"❌ Missing 'movieId' column in one of the files!\n\n"
                 f"Movies columns: {movies.columns.tolist()}\n"
                 f"Ratings columns: {ratings.columns.tolist()}")
        st.stop()

    # Merge datasets
    data = pd.merge(movies, ratings, on="movieId")

    # Create user–movie rating matrix
    pivot = data.pivot_table(index='userId', columns='title', values='rating').fillna(0)

    # Compute similarity
    similarity = cosine_similarity(pivot.T)
    sim_df = pd.DataFrame(similarity, index=pivot.columns, columns=pivot.columns)
    return sim_df

# Load the similarity matrix
similarity_df = load_data()

# --------------------- Streamlit App ---------------------
st.title("🎬 Movie Recommender App")

movie_name = st.selectbox("Choose a movie you like:", sorted(similarity_df.columns))

if st.button("Recommend 🎥"):
    st.subheader(f"Because you liked **{movie_name}**, you might also enjoy:")
    similar_movies = similarity_df[movie_name].sort_values(ascending=False)[1:6]
    for i, (movie, score) in enumerate(similar_movies.items(), start=1):
        st.write(f"{i}. **{movie}**  (Similarity: {score:.2f})")
