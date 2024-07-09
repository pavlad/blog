from pathlib import Path
import streamlit as st

# --- PATH SETTINGS ---
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file = current_dir / "styles" / "main.css"
resume_file = current_dir / "assets" / "CV.pdf"


# --- GENERAL SETTINGS ---
PAGE_TITLE = "Vladimir Pavlyukov"
PAGE_ICON = "🧩"
NAME = "Vladimir<br>Pavlyukov"
DESCRIPTION = """
Data Scientist | Software Engineer
"""

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, initial_sidebar_state='collapsed')


# --- LOAD CSS, PDF & PROFIL PIC ---
with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

# --- HERO SECTION ---
with st.container():
    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: center; gap: 40px;">
      <div style="display: flex; align-items: center; height: 100%; justify-content: flex-end;">
        <img src="app/static/profile-pic.png" style="max-height: 150px; object-fit: contain;" />
      </div>
      <div style="display: flex; flex-direction: column; justify-content: center;">
        <h1>{NAME}</h1>
        <p>{DESCRIPTION}</p>
      </div>
    </div>
    """, unsafe_allow_html=True)


st.write('\n')
# Inject custom CSS
st.markdown(
    """
    <style>
    .custom-h2 {
        font-size: 1.5em; /* Adjust the size as needed */
        font-weight: 300; /* Adjust the weight as needed */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Use the custom class for your text
st.markdown("""
<p class="custom-h2">With seven years of software engineering experience and a recent BSc in Data Science, I've developed a strong passion for machine learning, particularly in NLP. <br>This site is a space to share my learnings and showcase projects.</p>
""", unsafe_allow_html=True)

st.write('\n')
st.subheader("Projects")
if st.button('Transformer architecture from scratch in TensorFlow', use_container_width=True, type='primary'):
    st.switch_page("pages/Transformer from scratch in TF.py")
if st.button('Named Entity Recognition with BERT', use_container_width=True, type='primary'):
    st.switch_page("pages/NER demo.py")
st.button('Topic modeling with BERTopic', use_container_width=True, type='primary')

# --- EXPERIENCE & QUALIFICATIONS ---
st.write('\n')
st.subheader("Background")
st.write(
    """
- 7 years of experience as software engineer
- BSc in Data Science
- Proficient in NLP techniques and frameworks
- Team player with strong sense of ownership
"""
)


# --- SKILLS ---
st.write('\n')
st.subheader("Skills")
st.write("""
- **Data Science**: Machine Learning (TensorFlow, Pandas, numpy, scikit-learn), NLP, linear regression, K-means, k-nearest-neighbors, data analysis (cleaning, modelling, visualisation)
- **ML Engineering**: Docker Swarm, Kubernetes, SQL, NoSQL, ETL
- **Software Engineering**: Python, R, Ruby, JavaScript, TypeScript, GraphQL, Git, APIs, CI/CD, object oriented programming, testing
- **Languages**: Dutch, English, French, Russian
"""
)
