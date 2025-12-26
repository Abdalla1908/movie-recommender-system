import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import time
import os

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Define NLPRecommender class
class NLPRecommender:
    def __init__(self):
        self.df = None
        self.sbert_model = None
        self.sbert_matrix = None
        self.is_trained = False

    def train(self, df):
        """Train the model by generating SentenceTransformer embeddings for movie descriptions."""
        print("Training NLPRecommender model...")
        self.df = df.copy()
        
        # Preprocess text
        stop_words = set(stopwords.words('english'))
        def preprocess_text(text):
            if not isinstance(text, str):
                return ''
            tokens = word_tokenize(text.lower())
            tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
            return ' '.join(tokens)
        
        self.df['clean_description'] = self.df['description'].apply(preprocess_text)
        
        # Generate embeddings using SentenceTransformer
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.sbert_matrix = self.sbert_model.encode(self.df['clean_description'].tolist(), show_progress_bar=True)
        self.df['sbert_embedding'] = list(self.sbert_matrix)
        
        self.is_trained = True
        print("Training completed.")

    def recommend(self, movie_title=None, top_n=5, actor=None, min_rating=None, genre=None, age_rating=None):
        """Generate movie recommendations based on cosine similarity or filters."""
        if not self.is_trained:
            raise ValueError("Model is not trained. Call train() first.")
        
        filtered_df = self.df.copy()
        
        # Apply filters
        if actor:
            filtered_df = filtered_df[filtered_df['actors'].str.contains(actor, case=False, na=False)]
        if min_rating:
            filtered_df = filtered_df[filtered_df['rating'] >= min_rating]
        if genre:
            filtered_df = filtered_df[filtered_df['genre'].str.contains(genre, case=False, na=False)]
        if age_rating:
            filtered_df = filtered_df[filtered_df['age_rating'].str.contains(age_rating, case=False, na=False)]
        
        # If movie_title is provided, use cosine similarity
        if movie_title and movie_title in self.df['title'].values:
            idx = self.df.index[self.df['title'] == movie_title][0]
            
            # Calculate cosine similarity
            sbert_sim = cosine_similarity(self.sbert_matrix[idx].reshape(1, -1), self.sbert_matrix).flatten()
            
            # Create similarity DataFrame
            sim_df = pd.DataFrame({'sim_score': sbert_sim}, index=self.df.index)
            sim_df = sim_df.join(self.df[['title', 'rating', 'genre', 'actors', 'age_rating', 'poster']])
            
            # Get input movie
            input_movie = sim_df.loc[[idx]][['title', 'rating', 'genre', 'actors', 'age_rating', 'poster', 'sim_score']]
            
            # Filter and sort recommendations
            filtered_sim_df = sim_df.loc[filtered_df.index].drop(index=idx, errors='ignore')
            filtered_sim_df = filtered_sim_df.sort_values(by='sim_score', ascending=False)
            similar_movies = filtered_sim_df.head(top_n-1)[['title', 'rating', 'genre', 'poster', 'sim_score']]
            
            recommendations = pd.concat([input_movie, similar_movies])
            recommendations = recommendations.sort_values(by='sim_score', ascending=False).head(top_n)
            recommendations = recommendations[['title', 'rating', 'genre', 'poster']]
            
            if recommendations.empty:
                return "No movies match the specified filters."
            
            return recommendations
        
        # If no movie_title, return top-rated movies based on filters
        else:
            if filtered_df.empty:
                return "No movies match the specified filters."
            if not any([movie_title, actor, min_rating, genre, age_rating]):
                return "Please provide at least one filter to get recommendations."
            
            # Sort by rating (highest to lowest)
            recommendations = filtered_df.sort_values(by='rating', ascending=False).head(top_n)
            recommendations = recommendations[['title', 'rating', 'genre', 'poster']]
            
            if recommendations.empty:
                return "No movies match the specified filters."
            
            return recommendations

    def evaluate(self, top_n=5):
        """Evaluate the model using diversity and coverage metrics."""
        if not self.is_trained:
            raise ValueError("Model is not trained. Call train() first.")
        
        print("Evaluating NLPRecommender model...")
        
        # Intra-list Diversity: Average pairwise cosine similarity within recommended lists
        diversity_scores = []
        for idx in self.df.index:
            sim_scores = cosine_similarity(self.sbert_matrix[idx].reshape(1, -1), self.sbert_matrix).flatten()
            sim_df = pd.DataFrame({'sim_score': sim_scores}, index=self.df.index)
            sim_df = sim_df.sort_values(by='sim_score', ascending=False).head(top_n)
            if len(sim_df) > 1:
                rec_matrix = self.sbert_matrix[sim_df.index]
                pairwise_sim = cosine_similarity(rec_matrix)
                # Exclude diagonal (self-similarity)
                mask = np.ones_like(pairwise_sim, dtype=bool)
                np.fill_diagonal(mask, False)
                avg_pairwise_sim = pairwise_sim[mask].mean() if pairwise_sim[mask].size > 0 else 1.0
                diversity_scores.append(1 - avg_pairwise_sim)  # Higher diversity = lower similarity
        
        avg_diversity = np.mean(diversity_scores) if diversity_scores else 0.0
        
        # Coverage: Percentage of movies that can be recommended
        recommended_movies = set()
        for idx in self.df.index:
            sim_scores = cosine_similarity(self.sbert_matrix[idx].reshape(1, -1), self.sbert_matrix).flatten()
            sim_df = pd.DataFrame({'sim_score': sim_scores}, index=self.df.index)
            recommended_movies.update(sim_df.sort_values(by='sim_score', ascending=False).head(top_n).index)
        coverage = len(recommended_movies) / len(self.df) * 100
        
        # Visualize similarity score distribution
        all_sim_scores = []
        for idx in self.df.index:
            sim_scores = cosine_similarity(self.sbert_matrix[idx].reshape(1, -1), self.sbert_matrix).flatten()
            all_sim_scores.extend(sim_scores[sim_scores < 1.0])  # Exclude self-similarity
        plt.figure(figsize=(10, 6))
        sns.histplot(all_sim_scores, bins=50, kde=True)
        plt.title('Distribution of Cosine Similarity Scores', fontsize=14, weight='bold')
        plt.xlabel('Cosine Similarity', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.tight_layout()
        plt.savefig('similarity_distribution.png')
        plt.close()
        
        evaluation_results = {
            'avg_diversity': avg_diversity,
            'coverage': coverage,
            'n_movies': len(self.df)
        }
        
        print(f"Evaluation Results:")
        print(f"Average Intra-list Diversity: {avg_diversity:.4f} (higher is better)")
        print(f"Coverage: {coverage:.2f}% (percentage of movies recommended)")
        print(f"Total Movies: {evaluation_results['n_movies']}")
        
        return evaluation_results

# Load the trained model from pickle
MODEL_PATH = 'model.pkl'
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file '{MODEL_PATH}' not found. Please ensure the file exists in the project directory or update the MODEL_PATH in the code.")
    st.stop()
else:
    with open(MODEL_PATH, 'rb') as file:
        nlp_model = pickle.load(file)

# Custom CSS for styling
st.markdown("""
    <style>
    .stApp {
        background-image: url('background.png');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: white;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stTextInput, .stNumberInput, .stSelectbox {
        background-color: transparent;
        border: 1px solid #ffd700;
        border-radius: 5px;
        color: white;
        padding: 5px;
    }
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        color: white !important;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #ffd700;
        background-color: rgba(0, 0, 0, 0.5);
        padding: 5px 10px;
        border-radius: 5px;
        display: inline-block;
    }
    .movie-card {
        background-color: rgba(0, 0, 0, 0.5);
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stSpinner, .stProgress, .stSuccess, .stError, .stWrite {
        background-color: rgba(0, 0, 0, 0.5);
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Part 1: Web Scraping using TMDb API
def fetch_tmdb_movies(api_key, max_movies):
    movies_data = []
    base_url = "https://api.themoviedb.org/3/discover/movie"
    params = {
        'api_key': '92e89881fee050a9add5845a3f7a8223',
        'language': 'en-US',
        'sort_by': 'popularity.desc',
        'page': 1
    }
    
    movie_count = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while movie_count < max_movies:
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            st.error(f"Failed to fetch data from TMDb: {response.status_code}")
            break
        
        data = response.json()
        for movie in data['results']:
            if movie_count >= max_movies:
                break
            movie_id = movie['id']
            details_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
            credits_url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits"
            details_params = {'api_key': '92e89881fee050a9add5845a3f7a8223', 'language': 'en-US'}
            
            details_response = requests.get(details_url, params=details_params)
            credits_response = requests.get(credits_url, params=details_params)
            
            if details_response.status_code == 200 and credits_response.status_code == 200:
                details = details_response.json()
                credits = credits_response.json()
                
                directors = [crew['name'] for crew in credits.get('crew', []) if crew.get('job') == 'Director']
                director = ', '.join(directors) if directors else 'N/A'
                
                movies_data.append({
                    'title': movie['title'],
                    'rating': movie['vote_average'],
                    'genre': ', '.join([g['name'] for g in details.get('genres', [])]),
                    'description': movie['overview'],
                    'actors': ', '.join([c['name'] for c in credits.get('cast', [])[:3]]),
                    'director': director,
                    'age_rating': details.get('release_dates', {}).get('results', [{}])[0].get('release_dates', [{}])[0].get('certification', 'N/A'),
                    'poster': f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie['poster_path'] else 'N/A',
                    'release_year': details.get('release_date', 'N/A')[:4]
                })
                movie_count += 1
                
                # Update progress
                progress = min(movie_count / max_movies, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Fetched {movie_count} of {max_movies} movies...")
        
        params['page'] += 1
        time.sleep(0.1)  # To avoid hitting API rate limits
    
    progress_bar.empty()
    status_text.empty()
    
    df = pd.DataFrame(movies_data)
    df = df.drop_duplicates(subset=['title'])
    df = df.reset_index(drop=True)
    if not df.empty:
        df.to_csv('movies_data.csv', index=False)
    return df

# Part 2: Text Representation (Update model data if necessary)
def text_representation(df, nlp_model):
    # Update the model's DataFrame if it's different
    if nlp_model.df is None or not nlp_model.df.equals(df):
        print("Updating NLPRecommender with new data...")
        nlp_model.train(df)
    return nlp_model

# Part 3: Recommendation System
def recommend_movies(nlp_model, top_n=5, actor=None, min_rating=None, genre=None, age_rating=None, title=None):
    recommendations = nlp_model.recommend(
        movie_title=title if title and title.strip() else None,
        top_n=top_n,
        actor=actor,
        min_rating=min_rating,
        genre=genre,
        age_rating=age_rating
    )
    
    return recommendations

# Streamlit App
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select Page", ["Data Collection", "Dashboards and Explore Movies", "Recommendations"])

# Initialize session state
if 'movies_df' not in st.session_state:
    st.session_state['movies_df'] = None
    st.session_state['nlp_model'] = None

# Load API key
API_KEY = '92e89881fee050a9add5845a3f7a8223'  # Replace with your TMDb API key

# Page 1: Data Collection
if page == "Data Collection":
    st.title("Movie Data Collection")
    st.markdown("**Enter the number of movies to fetch**")
    max_movies = st.number_input("", min_value=10, value=100, step=10, label_visibility="collapsed")
    if max_movies > 500:
        st.warning("Fetching a large number of movies (e.g., thousands) may take a long time due to API rate limits and processing requirements. Ensure you have a stable internet connection and sufficient memory.")
    if st.button("Start Data Collection"):
        with st.spinner(f"Fetching {max_movies} movies... This may take a while for large numbers."):
            movies_df = fetch_tmdb_movies(API_KEY, max_movies)
            if not movies_df.empty:
                nlp_model = text_representation(movies_df, nlp_model)
                st.session_state['movies_df'] = movies_df
                st.session_state['nlp_model'] = nlp_model
                st.success(f"Successfully collected {len(movies_df)} movies!")
            else:
                st.error("Failed to collect data. Check your API key or internet connection.")

# Page 2: Dashboards and Explore Movies
elif page == "Dashboards and Explore Movies":
    st.title("Dashboards and Explore Movies")
    
    if st.session_state.get('movies_df') is None:
        if os.path.exists('movies_data.csv'):
            try:
                movies_df = pd.read_csv('movies_data.csv')
                if movies_df.empty:
                    st.error("movies_data.csv is empty. Please collect data from the Data Collection page.")
                else:
                    nlp_model = text_representation(movies_df, nlp_model)
                    st.session_state['movies_df'] = movies_df
                    st.session_state['nlp_model'] = nlp_model
            except Exception as e:
                st.error(f"Error loading movies_data.csv: {str(e)}")
        else:
            st.error("Please collect data first from the Data Collection page.")
    else:
        movies_df = st.session_state['movies_df']
        
        # Dashboards
        st.header("Analytics Dashboard")
        st.subheader("Genre Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        genres = movies_df['genre'].str.split(', ').explode().dropna()
        genres = genres.reset_index(drop=True)
        sns.countplot(y=genres, order=genres.value_counts().index[:10], palette='viridis', ax=ax)
        ax.set_title('Top 10 Most Common Genres', fontsize=14, weight='bold')
        ax.set_xlabel('Count', fontsize=12)
        ax.set_ylabel('Genre', fontsize=12)
        st.pyplot(fig)
        plt.close(fig)
        
        st.subheader("Rating Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(movies_df['rating'].dropna(), bins=20, color='coral', edgecolor='black', ax=ax)
        ax.set_title('Distribution of Movie Ratings', fontsize=14, weight='bold')
        ax.set_xlabel('Rating', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        st.pyplot(fig)
        plt.close(fig)
        
        st.subheader("Description Word Cloud")
        fig, ax = plt.subplots(figsize=(10, 6))
        text = ' '.join(movies_df['description'].dropna())
        wordcloud = WordCloud(width=800, height=400, max_words=100, background_color='white', colormap='plasma').generate(text)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Word Cloud of Movie Descriptions', fontsize=14, weight='bold')
        st.pyplot(fig)
        plt.close(fig)
        
        st.subheader("Ratings vs. Release Year")
        fig, ax = plt.subplots(figsize=(10, 6))
        movies_df['release_year'] = pd.to_numeric(movies_df['release_year'], errors='coerce')
        sns.scatterplot(x='release_year', y='rating', data=movies_df.dropna(subset=['release_year', 'rating']), hue='rating', palette='coolwarm', size='rating', ax=ax)
        ax.set_title('Ratings vs. Release Year', fontsize=14, weight='bold')
        ax.set_xlabel('Release Year', fontsize=12)
        ax.set_ylabel('Rating', fontsize=12)
        st.pyplot(fig)
        plt.close(fig)
        
        st.subheader("Age Rating Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='age_rating', data=movies_df, order=movies_df['age_rating'].value_counts().index[:5], palette='magma', ax=ax)
        ax.set_title('Distribution of Age Ratings', fontsize=14, weight='bold')
        ax.set_xlabel('Age Rating', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        st.pyplot(fig)
        plt.close(fig)
        
        st.subheader("Top 10 Most Frequent Actors")
        fig, ax = plt.subplots(figsize=(10, 6))
        actors_list = movies_df['actors'].str.split(', ', expand=True).stack().value_counts().head(10)
        actors_list.plot(kind='bar', color='skyblue', ax=ax)
        ax.set_title('Top 10 Most Frequent Actors', fontsize=14, weight='bold')
        ax.set_xlabel('Actor', fontsize=12)
        ax.set_ylabel('Number of Movies', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        st.subheader("Top 10 Most Prolific Directors")
        fig, ax = plt.subplots(figsize=(10, 6))
        directors_list = movies_df['director'].str.split(', ', expand=True).stack().value_counts().head(10)
        directors_list.plot(kind='bar', color='lightgreen', ax=ax)
        ax.set_title('Top 10 Most Prolific Directors', fontsize=14, weight='bold')
        ax.set_xlabel('Director', fontsize=12)
        ax.set_ylabel('Number of Movies', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        st.subheader("Release Year Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(movies_df['release_year'].dropna(), bins=20, kde=True, color='orange', ax=ax)
        ax.set_title('Distribution of Release Years', fontsize=14, weight='bold')
        ax.set_xlabel('Release Year', fontsize=12)
        ax.set_ylabel('Number of Movies', fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        st.subheader("Rating Distribution by Genre")
        genre_df = movies_df.assign(genre=movies_df['genre'].str.split(', ')).explode('genre')
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(x='genre', y='rating', data=genre_df, palette='Set2', ax=ax)
        ax.set_title('Rating Distribution by Genre', fontsize=14, weight='bold')
        ax.set_xlabel('Genre', fontsize=12)
        ax.set_ylabel('Rating', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Explore Movies
        st.header("Explore Movies")
        unique_genres = [''] + sorted(set(genre for genre in movies_df['genre'].str.split(', ').explode().dropna()))
        unique_age_ratings = [''] + sorted([rating for rating in movies_df['age_rating'].dropna().unique()])
        
        st.markdown("**Select Genre**")
        genre_filter = st.selectbox("", unique_genres, label_visibility="collapsed")
        st.markdown("**Minimum Rating**")
        min_rating_filter = st.number_input("", min_value=0.0, max_value=10.0, value=0.0, label_visibility="collapsed")
        st.markdown("**Select Age Rating**")
        age_rating_filter = st.selectbox("", unique_age_ratings, label_visibility="collapsed")
        
        filtered_df = movies_df.copy()
        if genre_filter:
            filtered_df = filtered_df[filtered_df['genre'].str.contains(genre_filter, case=False, na=False)]
        if min_rating_filter > 0.0:
            filtered_df = filtered_df[filtered_df['rating'] >= min_rating_filter]
        if age_rating_filter:
            filtered_df = filtered_df[filtered_df['age_rating'].str.contains(age_rating_filter, case=False, na=False)]
        
        st.write(f"**Number of Matching Movies**: {len(filtered_df)}")
        for idx, row in filtered_df.iterrows():
            st.markdown(f"<div class='movie-card'>", unsafe_allow_html=True)
            if row['poster'] != 'N/A':
                try:
                    response = requests.get(row['poster'].replace('/w500/', '/w185/'))
                    img = Image.open(BytesIO(response.content))
                    st.image(img, caption=row['title'], width=150)
                except:
                    st.write(f"{row['title']} (Image not available)")
            else:
                st.write(f"{row['title']} (Image not available)")
            st.write(f"**Rating**: {row['rating']}, **Genre**: {row['genre']}, **Age Rating**: {row['age_rating']}")
            st.markdown("</div>", unsafe_allow_html=True)

# Page 3: Recommendations
elif page == "Recommendations":
    st.title("Recommendations")
    
    if st.session_state.get('movies_df') is None:
        if os.path.exists('movies_data.csv'):
            try:
                movies_df = pd.read_csv('movies_data.csv')
                if movies_df.empty:
                    st.error("movies_data.csv is empty. Please collect data from the Data Collection page.")
                else:
                    nlp_model = text_representation(movies_df, nlp_model)
                    st.session_state['movies_df'] = movies_df
                    st.session_state['nlp_model'] = nlp_model
            except Exception as e:
                st.error(f"Error loading movies_data.csv: {str(e)}")
        else:
            st.error("Please collect data first from the Data Collection page.")
    else:
        movies_df = st.session_state['movies_df']
        nlp_model = st.session_state['nlp_model']
        
        st.header("Enter Filters")
        unique_genres = [''] + sorted(set(genre for genre in movies_df['genre'].str.split(', ').explode().dropna()))
        unique_age_ratings = [''] + sorted([rating for rating in movies_df['age_rating'].dropna().unique()])
        
        st.markdown("**Movie Title (Optional)**")
        title = st.text_input("", label_visibility="collapsed", key="title_filter", placeholder="Enter a movie title")
        st.markdown("**Actor Name (Optional)**")
        actor = st.text_input("", label_visibility="collapsed", key="actor_filter", placeholder="Enter an actor name")
        st.markdown("**Minimum Rating (Optional)**")
        min_rating = st.number_input("", min_value=0.0, max_value=10.0, value=0.0, label_visibility="collapsed")
        st.markdown("**Select Genre (Optional)**")
        genre = st.selectbox("", unique_genres, label_visibility="collapsed")
        st.markdown("**Select Age Rating (Optional)**")
        age_rating = st.selectbox("", unique_age_ratings, label_visibility="collapsed")
        
        if st.button("Get Recommendations"):
            recommendations = recommend_movies(
                nlp_model,
                top_n=5,
                actor=actor if actor else None,
                min_rating=min_rating if min_rating > 0.0 else None,
                genre=genre if genre else None,
                age_rating=age_rating if age_rating else None,
                title=title if title else None
            )
            
            if isinstance(recommendations, str):
                st.error(recommendations)
            else:
                st.header("Recommended Movies")
                for idx, row in recommendations.iterrows():
                    st.markdown(f"<div class='movie-card'>", unsafe_allow_html=True)
                    if row['poster'] != 'N/A':
                        try:
                            response = requests.get(row['poster'].replace('/w500/', '/w185/'))
                            img = Image.open(BytesIO(response.content))
                            st.image(img, caption=row['title'], width=150)
                        except:
                            st.write(f"{row['title']} (Image not available)")
                    else:
                        st.write(f"{row['title']} (Image not available)")
                    st.write(f"**Rating**: {row['rating']}, **Genre**: {row['genre']}")
                    st.markdown("</div>", unsafe_allow_html=True)