import streamlit as st
import sys

sys.path.insert(1, "/opt/homebrew/lib/python3.11/site-packages")


from streamlit_option_menu import option_menu

with st.sidebar:
    selected = option_menu("Main Menu", ["Home", 'Settings'], 
        icons=['house', 'gear'], menu_icon="cast", default_index=1)
    selected