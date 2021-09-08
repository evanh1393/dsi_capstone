import streamlit as st
import pandas as pd

# Custom imports
from multipage import MultiPage
from pages import new_users, cb_engine, cf_engine # import your pages here
# Create an instance of the app
app = MultiPage()

# Add all your applications (pages) here
app.add_page('New User', new_users.app)
app.add_page('Content Based Engine', cb_engine.app)
app.add_page('Collaborative Filtering Engine', cf_engine.app)


# The main app
app.run()
