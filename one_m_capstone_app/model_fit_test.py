import pandas as pd
import numpy as np

import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

ratings = pd.read_csv('./data/combined.csv')
app_ratings = pd.read_csv('./app_data/new_ratings.csv')

ratings = ratings[['user_id','movie_id','rating']]
app_ratings = app_ratings[['user_id','movie_id','rating']]

df = pd.concat([ratings, app_ratings])

user_enc = LabelEncoder()
df['user'] = user_enc.fit_transform(df['user_id'].values)
n_users = df['user'].nunique()

item_enc = LabelEncoder()
df['movie'] = item_enc.fit_transform(df['movie_id'].values)
n_movies = df['movie'].nunique()

df['rating'] = df['rating'].values.astype(np.float32)  # make it more workable with keras
X = df[['user', 'movie']].values
y = df["rating"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_array = [X_train[:, 0], X_train[:, 1]]
X_test_array = [X_test[:, 0], X_test[:, 1]]
model = tf.keras.models.load_model('model.keras')
model.fit(x=X_train_array, y=y_train,
                    batch_size=128,
                    epochs=2,
                    verbose=1,
                    validation_data=(X_test_array,y_test))

tf.keras.models.save_model(model, 'model.keras')
pickle.dump(user_enc, open('user_enc.pkl','wb'))
pickle.dump(item_enc, open('item_enc.pkl','wb'))
