import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer



class Recommender:
    def __init__(self, df):
        self.df = df.copy()
        self.desc_embeddings = None
        self.genre_tfidf = None
        self.cast_tfidf = None
        self.similarity_matrix = None
        self.processed_df = None
        
    def normalize_token(self, token):
        if not isinstance(token, str):
            return ""
        token = token.lower()
        token = re.sub(r'[\s\-_]+', '', token)     # remove spaces, - , _
        token = re.sub(r'[^a-z0-9]', '', token)    # remove anything else
        return token
    
    def process_genres(self):
        self.df['genres_list'] = self.df['genres'].apply(lambda x: str(x))
        self.df['genres_list'] = self.df['genres_list'].apply(lambda x: [self.normalize_token(genre) for genre in x.split(', ')])
        self.df['genres_list'] = self.df['genres_list'].apply(lambda lst: " ".join(x for x in lst)
                                            if isinstance(lst, list) else "")  # convert the list to one string
        return self.df['genres_list']

    def process_cast(self):
        self.df['cast_list'] = self.df['cast'].apply(lambda x: str(x))
        self.df['cast_list'] = self.df['cast_list'].apply(lambda x: [self.normalize_token(actor) for actor in x.split(', ')])
        self.df['cast_list'] = self.df['cast_list'].apply(lambda lst: " ".join(x for x in lst)
                                            if isinstance(lst, list) else "")  # convert the list to one string

        return self.df['cast_list']
    
    def clean_description(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'<.*?>', ' ', text)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def generate_description_embeddings(self, column="description"):
        self.df["clean_description"] = self.df[column].apply(self.clean_description)
        
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(
            self.df["clean_description"].tolist(),
            show_progress_bar=True
        )
        
        return embeddings, self.df
    
    def tfidf_vectorization(self, column):
        self.df = self.df.copy()
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english'
        )
        tfidf_matrix = vectorizer.fit_transform(self.df[column].fillna(""))
        
        return tfidf_matrix, self.df
    
    def combined_similarity(
        self,
        desc_emb,
        genre_tfidf,
        cast_tfidf,
        w_desc=0.7,
        w_genre=0.2,
        w_cast=0.1
    ):
        sim_desc = cosine_similarity(desc_emb)
        sim_genre = cosine_similarity(genre_tfidf)
        sim_cast = cosine_similarity(cast_tfidf)
        
        final_sim = (
            w_desc * sim_desc +
            w_genre * sim_genre +
            w_cast * sim_cast
        )
        
        return final_sim
    
    def fit(self, w_desc=0.7, w_genre=0.2, w_cast=0.1):
        # Process genres and cast
        self.process_genres()
        self.process_cast()
        
        # Generate description embeddings
        self.desc_embeddings, self.processed_df = self.generate_description_embeddings()
        
        # Generate TF-IDF matrices
        self.genre_tfidf, _ = self.tfidf_vectorization('genres_list')
        self.cast_tfidf, _ = self.tfidf_vectorization('cast_list')
        
        # Calculate combined similarity matrix
        self.similarity_matrix = self.combined_similarity(
            self.desc_embeddings,
            self.genre_tfidf,
            self.cast_tfidf,
            w_desc, w_genre, w_cast
        )
        
        return self
    
    def recommend(self, movie_title, num_recommendations=5):
        if self.similarity_matrix is None:
            raise ValueError("Model not fitted yet. Call fit() method first.")
        
        # Find the index of the movie
        try:
            movie_idx = self.processed_df[self.processed_df['title'].str.lower() == movie_title.lower()].index[0]
        except IndexError:
            # If exact match not found, try partial match
            movie_indices = self.processed_df[self.processed_df['title'].str.lower().str.contains(movie_title.lower())].index
            if len(movie_indices) == 0:
                raise ValueError(f"Movie '{movie_title}' not found in the dataset.")
            movie_idx = movie_indices[0]  # Take the first match
        
        # Get similarity scores for the movie
        sim_scores = list(enumerate(self.similarity_matrix[movie_idx]))
        
        # Sort movies based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get the indices of most similar movies
        movie_indices = [i[0] for i in sim_scores[1:num_recommendations+1]]  # Exclude the movie itself
        
        # Return the recommended movies
        recommendations = self.processed_df.iloc[movie_indices][['title', 'genres', 'cast', 'description', 'poster']].copy()
        recommendations['similarity_score'] = [sim_scores[i+1][1] for i in range(len(movie_indices))]
        
        return recommendations
    
    def evaluate(self, test_movies=None, num_recommendations=5):
        """
        Evaluate the recommender system by calculating precision and recall
        """
        if test_movies is None:
            # Use a random sample of movies for evaluation
            test_movies = self.processed_df.sample(min(10, len(self.processed_df)))['title'].tolist()
        
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        valid_evaluations = 0
        
        for movie in test_movies:
            try:
                recommendations = self.recommend(movie, num_recommendations)
                
                # For evaluation, we can calculate how diverse the recommendations are
                # or how well they match the original movie's genre/cast
                original_movie = self.processed_df[self.processed_df['title'].str.lower() == movie.lower()]
                if original_movie.empty:
                    continue
                
                original_genres = set(original_movie.iloc[0]['genres'].lower().split(', '))
                original_cast = set(original_movie.iloc[0]['cast'].lower().split(', '))
                
                # Calculate how many recommended movies share genres or cast with original
                relevant_recommendations = 0
                for _, rec in recommendations.iterrows():
                    rec_genres = set(rec['genres'].lower().split(', '))
                    rec_cast = set(rec['cast'].lower().split(', '))
                    
                    # Check if there's overlap in genres or cast
                    if original_genres.intersection(rec_genres) or original_cast.intersection(rec_cast):
                        relevant_recommendations += 1
                
                # Calculate metrics
                precision = relevant_recommendations / num_recommendations if num_recommendations > 0 else 0
                recall = relevant_recommendations / min(num_recommendations, len(self.processed_df)-1) if len(self.processed_df) > 1 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                total_precision += precision
                total_recall += recall
                total_f1 += f1
                valid_evaluations += 1
                
            except Exception as e:
                print(f"Error evaluating movie '{movie}': {e}")
                continue
        
        if valid_evaluations > 0:
            avg_precision = total_precision / valid_evaluations
            avg_recall = total_recall / valid_evaluations
            avg_f1 = total_f1 / valid_evaluations
            
            evaluation_results = {
                'avg_precision': avg_precision,
                'avg_recall': avg_recall,
                'avg_f1_score': avg_f1,
                'num_evaluated': valid_evaluations,
                'total_test_movies': len(test_movies)
            }
            
            return evaluation_results
        
