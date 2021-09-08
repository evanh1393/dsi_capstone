import pandas as pd
import numpy as np
import streamlit as st
import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

@st.cache
def load_df():
    return pd.read_csv('./data/combined.csv')
df = load_df()

##############
# Functions
##############
@st.cache
def load_movies():
    return pd.read_csv('./data/processed_movies.csv').drop(columns=['Unnamed: 0'])

def load_users():
    return pd.read_csv('./app_data/app_users.csv')

def load_ratings():
    return pd.read_csv('./app_data/new_ratings.csv')

def get_random_movies(fav_genres):
  # takes in favorite genres and returns a randomly selected choice of movies
  # from the user's favorite genres
  rand_movies = []

  # Find one random movie from each of the favorite genres that shares the genres
  for genre in fav_genres:
    genre_movies = movies[movies['genres'] == genre]['title']
    rand_movies.append(genre_movies.sample().values[0])

  return rand_movies



def app():
    movies = load_movies()
    app_users = load_users()
    ratings_df = load_ratings()


    # page title
    st.markdown('# Content Recommendation Engine')

    # maps usernames to id
    user_id_map = dict(zip(app_users['user_name'],app_users['user_id']))

    selected_user = st.selectbox('Select a user to make recommendations for', user_id_map.keys(), index=0)

    # select the user id to update on back refrencing the map
    user = user_id_map[selected_user]

    fav_genres = app_users.query(f'user_id == {user}')['genres'].values[0].split(' ')

    # grab 3 random genres
    fav_genres = np.random.choice(fav_genres, 3, replace=False)
    fav_genres = sorted(fav_genres)
    # id to title map
    title_map = dict(zip(movies['movie_id'].values,movies['title'].values))


    # Breaking down the inputs to speed up the Ratings
    # starting by ordering all the movies by popularity
    popularity_list = df.groupby('movie_id')['rating'].sum().sort_values(ascending=False).keys()

    seen_movies = []
    # check if user is in the system. if so eliminate seen movies from movie list
    if ratings_df.query(f'user_id=={user}').shape[0] != 0:
        seen_movies = ratings_df.query(f'user_id=={user}')['movie_id'].values

    unseen_movies = [x for x in popularity_list if x not in seen_movies]

    # id to genre map
    idgenre_map = dict(zip(movies['movie_id'],movies['genres']))

    genre_one_movies = [x for x in unseen_movies if fav_genres[0] in idgenre_map[x]]
    genre_two_movies = [x for x in unseen_movies if fav_genres[1] in idgenre_map[x]]
    genre_two_movies = [x for x in genre_two_movies if x not in genre_one_movies]
    genre_three_movies = [x for x in unseen_movies if fav_genres[2] in idgenre_map[x]]
    genre_three_movies = [x for x in genre_three_movies if x not in genre_one_movies]
    genre_three_movies = [x for x in genre_three_movies if x not in genre_two_movies]

    # start with three popular movies
    popular_movies = {
        genre_one_movies[0] : title_map[genre_one_movies[0]],
        genre_two_movies[0] : title_map[genre_two_movies[0]],
        genre_three_movies[0] : title_map[genre_three_movies[0]]
    }

    title_to_ids = dict(zip(movies['title'],movies['movie_id']))

    # returns 2 like titles
    def genre_recommendations(title):
        # Create a tf-idf matrix of genres found
        tf = TfidfVectorizer()
        tfidf_matrix = tf.fit_transform(movies['genres'])

        # Cosine similarity is done implicitly by the l2 normalization applied from the tf-idf
        # matrix, we can just use linear_kernel
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        # Looks at a title and returns the top 5 most similar titles
        titles = movies['title']
        indices = pd.Series(movies.index, index=movies['title'])
        idx = indices[title]

        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        movie_indices = [i[0] for i in sim_scores]
        indicies = [id for id in titles.iloc[movie_indices].keys() if id not in seen_movies]
        return titles.iloc[indicies][:2]

    genre_similar = genre_recommendations(genre_one_movies[0])

    genre_similar = genre_similar.append(genre_recommendations(genre_two_movies[0]))
    genre_similar = genre_similar.append(genre_recommendations(genre_three_movies[0]))

    genre_similar = genre_similar.to_dict()

    movies_to_rate = {**popular_movies, **genre_similar}

    with st.form("rec_start_form"):
        st.markdown('# Starter Recommendations Form')
        ratings = {}
        for i, movie in movies_to_rate.items():
            ratings[i] = st.text_input('Enter a rating for %s (1-5)'%title_map[i], 4,key='%s'%movie)
        submitted = st.form_submit_button("Submit Initial Ratings")

    if submitted:
        try:
            with open('./app_data/new_ratings.csv', 'a+', newline='') as f:
                writer = csv.writer(f)
                for key, rating in ratings.items():
                    writer.writerow([user, int(key), rating])
        except Exception as e:
            st.write(e)

app()
