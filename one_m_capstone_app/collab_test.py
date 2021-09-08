import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime
import tensorflow as tf
import keras
import csv
from sklearn.preprocessing import LabelEncoder
import pickle

##############
# BASIC Functions
##############
@st.cache(allow_output_mutation=True)
def load_df():
    return pd.read_csv('./data/combined.csv')
df = load_df() # just once no need to reload every time

@st.cache
def load_movies():
    return pd.read_csv('./data/processed_movies.csv').drop(columns=['Unnamed: 0'])
movies = load_movies() # this will not change

def load_users():
    return pd.read_csv('./app_data/app_users.csv')

def load_ratings():
    return pd.read_csv('./app_data/new_ratings.csv')

# Load in the keras model
def load_model():
    return keras.models.load_model('model.keras')
model = load_model()



def app():
    # creating user encoder for model TESTING PURPOSES ONLY
    user_enc = pickle.load(open('./user_enc.pkl','rb'))
    # creating movie encoder for model
    item_enc = pickle.load(open('./item_enc.pkl','rb'))

    # lists and maps
    title_map = dict(zip(movies['movie_id'].values,movies['title'].values))

    app_users = load_users()
    ratings_df = load_ratings()

    df = pd.read_csv('./data/combined.csv')
    # ids sorted by popularity
    pop_ordered = list(df.groupby('movie_id')['rating'].sum().sort_values(ascending=False).keys())

    # page title
    st.markdown('# Collaborative Recommendation Engine')
    st.write(item_enc.classes_)
    # maps usernames to id
    user_id_map = dict(zip(app_users['user_name'],app_users['user_id']))
    selected_user = st.selectbox('Select a user to make recommendations for', user_id_map.keys(), index=0)

    # select the user id to update on back refrencing the map
    user = user_id_map[selected_user]
    st.write(user)
    user_encoded = user_enc.transform([user]) # encoding user variable for predictions

    seen_movies = []
    # check if user is in the system. if so eliminate seen movies from movie list
    if ratings_df.query(f'user_id=={user}').shape[0] != 0:
        seen_movies = ratings_df.query(f'user_id=={user}')['movie_id'].values

    # Breaking down the inputs to speed up the Ratings
    # starting by ordering all the movies by popularity
    popularity_list = [x for x in pop_ordered if x not in seen_movies]
    # random 50 of the top 200 most popular movies
    rng = np.random.default_rng()

    high_tier =  random.sample(popularity_list[:200], k=10)
    mid_tier = random.sample(popularity_list[200:2000], k=20)
    low_tier = random.sample(popularity_list[2000:], k=25)


    st.write(ratings_df[ratings_df['movie_id']==3455])
    # returns the top 3 most popular movies from a list
    # based on user preference
    def get_movie_recs(movie_list, user_encoded):
        predictions = {}
        for mid in movie_list:
            movie_encoded = item_enc.transform([mid])
            predictions[mid] = model([user_encoded,movie_encoded], training=False).numpy()[0][0]
        predictions = pd.Series(predictions).sort_values(ascending=True).sort_values(ascending=False).head(3)
        return {key:title_map[key] for key in predictions.keys()}

    # getting movies to movies_to_rate
    movies_to_rate = get_movie_recs(high_tier,user_encoded)
    movies_to_rate.update(get_movie_recs(mid_tier,user_encoded))
    movies_to_rate.update(get_movie_recs(low_tier,user_encoded))

    with st.form("rec_collab_form"):
        st.markdown('# Collaborative Recommendations Form')
        ratings = {}
        for i, movie in movies_to_rate.items():
            ratings[i] = st.text_input('Enter a rating for %s (1-5)'%movie, 4,key='%s'%i)
        submitted = st.form_submit_button("Submit Ratings")

    if submitted:
        try:
            with open('./app_data/new_ratings.csv', 'a+', newline='') as f:
                writer = csv.writer(f)
                for key, rating in ratings.items():
                    writer.writerow([int(user), int(key), rating])
            st.write('Rating Saved')
        except Exception as e:
            st.write(e)
app()
