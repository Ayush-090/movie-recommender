import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load and Prepare Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")

    # Combine important text features
    df['combined'] = (
        df['genres'].fillna('') + ' ' +
        df['description'].fillna('') + ' ' +
        df['director'].fillna('') + ' ' +
        df['cast'].fillna('')
    )

    # Convert text to feature vectors using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(df['combined'])

    # Compute cosine similarity matrix
    similarity = cosine_similarity(vectors)

    return df, similarity

# -----------------------------
# Recommendation Function
# -----------------------------
def recommend(movie_title, df, similarity, n=5):
    if movie_title not in df['title'].values:
        return []

    # Get index of the selected movie
    idx = df.index[df['title'] == movie_title][0]

    # Get pairwise similarity scores
    scores = list(enumerate(similarity[idx]))

    # Sort movies by similarity (excluding itself)
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]

    # Get recommended movies
    recommended = [(df.iloc[i]['title'], df.iloc[i]['genres'], df.iloc[i]['director'], score)
                   for i, score in sorted_scores]
    return recommended

# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")
    st.title("🎥 Movie Recommender System (Content-Based)")
    st.markdown("Find movies similar to your favorites based on **genre, plot, and cast**.")

    df, similarity = load_data()

    # Dropdown for movie selection
    movie_titles = df['title'].dropna().tolist()
    selected_movie = st.selectbox("Choose a movie:", sorted(movie_titles))

    # Recommend button
    if st.button("Recommend Movies 🎞️"):
        st.subheader(f"Because you liked **{selected_movie}**, you might also enjoy:")
        results = recommend(selected_movie, df, similarity)

        for i, (title, genre, director, score) in enumerate(results, start=1):
            st.markdown(f"""
            **{i}. {title}**  
            🎭 *Genre:* {genre}  
            🎬 *Director:* {director}  
            🔗 *Similarity Score:* {score:.2f}
            """)
            st.divider()

    st.markdown("---")
    st.caption("Developed by Ayush Bhardwaj & Rishikesh | Powered by Streamlit & scikit-learn")

if __name__ == "__main__":
    main()
