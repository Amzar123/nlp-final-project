"""Fact-check portal. Run from the project root:

    streamlit run app/portal.py
"""
from pathlib import Path

import streamlit as st

import ui

STATIC = Path(__file__).parent / "static"

st.set_page_config(page_title="factlens", page_icon=str(STATIC / "icon.svg"), layout="wide",
                   initial_sidebar_state="expanded")
st.logo(str(STATIC / "logo.svg"), icon_image=str(STATIC / "icon.svg"), size="large")
ui.inject_css()

pages = [
    st.Page("views/check.py", title="Check", icon=":material/fact_check:", default=True),
    st.Page("views/walkthrough.py", title="Step by step", icon=":material/route:", url_path="follow"),
    st.Page("views/library.py", title="Library", icon=":material/menu_book:", url_path="library"),
    st.Page("views/report.py", title="Accuracy", icon=":material/insights:", url_path="report"),
]
nav = st.navigation(pages, position="sidebar")
nav.run()
