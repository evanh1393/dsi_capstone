import pandas as pd
import streamlit as st
import csv

def get_new_user_id():
    app_users = pd.read_csv('./app_data/app_users.csv')
    return int(app_users.iloc[-1]['user_id']) + 1

# Get User Info First
with st.form("my_form"):
    st.markdown('# New User Info')

    # get their username to make it intuitively trackable
    username  = st.text_input('Enter in a username')

    # get age
    age = st.text_input('How old are you?')

    # getting gender
    gender = st.selectbox('Enter your gender.', ('Female', 'Male'))

    # getting job
    job = st.selectbox('What is your job?', (
        'K-12 student', 'homemaker', 'programmer', 'technician/engineer',
        'academic/educator', 'clerical/admin', 'self-employed',
        'other or not specified', 'executive/managerial',
        'college/grad student', 'writer', 'retired', 'scientist', 'artist',
        'customer service', 'sales/marketing', 'doctor/health care',
        'unemployed', 'lawyer', 'farmer', 'tradesman/craftsman'
        )
    )

    # get 3 favorite genres
    genres = st.multiselect(
        'What are your favorite genres?',
        ['Animation', "Children's", 'Comedy', 'Musical', 'Romance', 'Drama',
        'Action', 'Adventure', 'Fantasy', 'Sci-Fi', 'War', 'Crime',
        'Thriller', 'Western', 'Horror', 'Mystery', 'Documentary',
        'Film-Noir'],
    )

    # submit button
    submitted = st.form_submit_button("Save User Info")

genres = ' '.join(genres)


if submitted:
    try:
        with open('./app_data/app_users.csv', 'a+', newline='') as f:
            writer = csv.writer(f)
            new_user_submission = [username, get_new_user_id(), age, gender, job, genres]
            writer.writerow(new_user_submission)
        st.write('User Saved')
    except Exception as e:
        st.write(e)
