from pathlib import Path

import streamlit as st


# --- PATH SETTINGS ---
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file = current_dir / "styles" / "main.css"
resume_file = current_dir / "assets" / "CV.pdf"


# --- GENERAL SETTINGS ---
PAGE_TITLE = "Vladimir Pavlyukov"
PAGE_ICON = ":wave:"
NAME = "Vladimir Pavlyukov"
DESCRIPTION = """
Data Scientist | Software Engineer
"""
# EMAIL = "johndoe@email.com"
# SOCIAL_MEDIA = {
#     "YouTube": "https://youtube.com/c/codingisfun",
#     "LinkedIn": "https://linkedin.com",
#     "GitHub": "https://github.com",
#     "Twitter": "https://twitter.com",
# }
# PROJECTS = {
#     "🏆 Sales Dashboard - Comparing sales across three stores": "https://youtu.be/Sb0A9i6d320",
#     "🏆 Income and Expense Tracker - Web app with NoSQL database": "https://youtu.be/3egaMfE9388",
#     "🏆 Desktop Application - Excel2CSV converter with user settings & menubar": "https://youtu.be/LzCfNanQ_9c",
#     "🏆 MyToolBelt - Custom MS Excel add-in to combine Python & Excel": "https://pythonandvba.com/mytoolbelt/",
# }


st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)


# --- LOAD CSS, PDF & PROFIL PIC ---
with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

# --- HERO SECTION ---
with st.container():
    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: center; gap: 40px;">
      <div style="display: flex; align-items: center; height: 100%; width: 100%; justify-content: flex-end;">
        <img src="app/static/profile-pic.png" style="max-height: 150px; object-fit: contain;" />
      </div>
      <div style="display: flex; flex-direction: column; justify-content: center; width: 100%;">
        <h1>{NAME}</h1>
        <p>{DESCRIPTION}</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

# # --- SOCIAL LINKS ---
# st.write('\n')
# cols = st.columns(len(SOCIAL_MEDIA))
# for index, (platform, link) in enumerate(SOCIAL_MEDIA.items()):
#     cols[index].write(f"[{platform}]({link})")

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
st.button('Transformer architecture from scratch in TensorFlow', use_container_width=True, type='primary')
st.button('Named Entity Recognition with BERT', use_container_width=True, type='primary')
st.button('Topic modeling with BERTopic', use_container_width=True, type='primary')

# --- EXPERIENCE & QUALIFICATIONS ---
st.write('\n')
st.subheader("Background")
st.write(
    """
- ✔️ 7 years of experience as software engineer
- ✔️ BSc in Data Science
- ✔️ Proficient in NLP techniques and frameworks
- ✔️ Team player with strong sense of ownership
"""
)


# --- SKILLS ---
st.write('\n')
st.subheader("Hard skills")
st.write(
    """
- 👩‍💻 Programming: Python (Scikit-learn, Pandas), SQL, VBA
- 📊 Data Visulization: PowerBi, MS Excel, Plotly
- 📚 Modeling: Logistic regression, linear regression, decition trees
- 🗄️ Databases: Postgres, MongoDB, MySQL
"""
)


# # --- WORK HISTORY ---
# st.write('\n')
# st.subheader("Work History")
# st.write("---")
#
# # --- JOB 1
# st.write("🚧", "**Senior Data Analyst | Ross Industries**")
# st.write("02/2020 - Present")
# st.write(
#     """
# - ► Used PowerBI and SQL to redeﬁne and track KPIs surrounding marketing initiatives, and supplied recommendations to boost landing page conversion rate by 38%
# - ► Led a team of 4 analysts to brainstorm potential marketing and sales improvements, and implemented A/B tests to generate 15% more client leads
# - ► Redesigned data model through iterations that improved predictions by 12%
# """
# )
#
# # --- JOB 2
# st.write('\n')
# st.write("🚧", "**Data Analyst | Liberty Mutual Insurance**")
# st.write("01/2018 - 02/2022")
# st.write(
#     """
# - ► Built data models and maps to generate meaningful insights from customer data, boosting successful sales eﬀorts by 12%
# - ► Modeled targets likely to renew, and presented analysis to leadership, which led to a YoY revenue increase of $300K
# - ► Compiled, studied, and inferred large amounts of data, modeling information to drive auto policy pricing
# """
# )
#
# # --- JOB 3
# st.write('\n')
# st.write("🚧", "**Data Analyst | Chegg**")
# st.write("04/2015 - 01/2018")
# st.write(
#     """
# - ► Devised KPIs using SQL across company website in collaboration with cross-functional teams to achieve a 120% jump in organic traﬃc
# - ► Analyzed, documented, and reported user survey results to improve customer communication processes by 18%
# - ► Collaborated with analyst team to oversee end-to-end process surrounding customers' return data
# """
# )
#
#
# # --- Projects & Accomplishments ---
# st.write('\n')
# st.subheader("Projects & Accomplishments")
# st.write("---")
# for project, link in PROJECTS.items():
#     st.write(f"[{project}]({link})")
