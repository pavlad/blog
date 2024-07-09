import streamlit as st
from pathlib import Path

st.set_page_config(initial_sidebar_state='expanded')

current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
parent_dir = current_dir.parent
css_file = parent_dir / "styles" / "main.css"

# --- LOAD CSS, PDF & PROFIL PIC ---
with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

# Read the markdown file
def read_markdown_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Load and display the markdown file content
markdown_content = read_markdown_file(f"{current_dir}/ner.md")
st.markdown(markdown_content, unsafe_allow_html=True)

if st.button("🤖 For a demo of the NER model, click here"):
    st.switch_page("pages/NER demo.py")