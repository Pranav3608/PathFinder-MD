import streamlit as st
from components.upload import render_uploader
from components.history_download import render_history_download
from components.ChatUI import render_chat

st.set_page_config(page_title = "PathFinder M.D", layout="wide")
st.title("PathFinder M.D")

render_uploader()
render_chat()
render_history_download()