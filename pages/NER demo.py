import streamlit as st
import requests
import html
import os
from pathlib import Path
from dotenv import load_dotenv

st.set_page_config(initial_sidebar_state='expanded')

current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
parent_dir = current_dir.parent
css_file = parent_dir / "styles" / "main.css"

# --- LOAD CSS, PDF & PROFIL PIC ---
with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

DEFAULT_TEXT = f"""De Limburgse bedrijven exporteerden de voorbije 6 maanden samen ruim 550 miljoen euro aan producten naar landen buiten de EU. Dat is een stijging met 4 procent in vergelijking met dezelfde periode vorig jaar. In heel Vlaanderen is de export in het eerste kwartaal dan weer gedaald met 9 procent in vergelijking met dezelfde periode vorig jaar.

"Dat komt enerzijds door de dalende impact van de hoge energie- en grondstofprijzen", zegt Christine Thonnon, manager internationalisatie bij Voka Limburg. "Daarnaast merken we dat er weer meer vertrouwen is bij de ondernemers om aan verre export te doen. Daarbij kijken ze ook naar nieuwe markten om aan risicospreiding te doen."

(vrt.be)
"""

# Load environment variables from .env file
load_dotenv()

# Retrieve the Hugging Face API key from environment variables
HUGGINGFACE_KEY = os.getenv("HUGGINGFACE_KEY")

# Define the API URL and headers
API_URL = "https://api-inference.huggingface.co/models/pavlad/bert-finetuned-ner-nl-wiki"
headers = {"Authorization": f"Bearer {HUGGINGFACE_KEY}"}


# Function to query the API
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


# Function to generate HTML for entity badges
def map_entity_group(entity_group):
    if entity_group == "PER":
        return "Person"
    elif entity_group == "LOC":
        return "Location"
    elif entity_group == "ORG":
        return "Organization"
    else:
        return entity_group


# Function to generate HTML for entity badges
def generate_entity_badge(entity):
    print('entity', entity)
    entity_group = map_entity_group(entity['entity_group'])
    word = html.escape(entity['word'])
    score = round(entity['score'], 3)

    color = ''
    if entity_group == 'Person':
        color = '#FFD700'  # gold
    elif entity_group == 'Location':
        color = '#1E90FF'  # dodger blue
    elif entity_group == 'Organization':
        color = '#FF6347'  # tomato

    badge_html = f"""
    <span style='background-color: {color}; border-radius: 5px; padding: 2px 5px; margin: 2px; display: inline-block; font-size: 14px;'>
        <strong>{entity_group}</strong>: {word} ({score})
    </span>
    """
    return badge_html


# Function to integrate badges into a bullet point list
def generate_entity_list(entities):
    print(entities)
    list_html = "<div>"
    for entity in entities:
        badge_html = generate_entity_badge(entity)
        list_html += f"<div>{badge_html}</div>"
    list_html += "</div>"
    return list_html


# Streamlit app
# st.title("")
st.markdown("""
# Named Entity Recognition demo

NER is a use of Natural Language Processing to categorize entities like people, locations, and organizations from unstructured text data. As part of wanting to learn more about the applications of BERT architecture, I've finetuned a [Dutch BERT model](https://huggingface.co/GroNLP/bert-base-dutch-cased) with a [Wikipedia NER dataset](https://huggingface.co/datasets/unimelb-nlp/wikiann).
""")
if st.button("📖 For a deeper dive into the training process, click here"):
    st.switch_page("pages/NER with BERT.py")

# Text input form
input_text = st.text_area('Text to analyse:', DEFAULT_TEXT, height=350)

# Button to submit the form
if st.button("🔍 Analyse"):
    output = query({"inputs": input_text})

    # Check if the model is still loading
    while 'error' in output and 'loading' in output['error']:
        estimated_time = output.get('estimated_time', 20.0)
        st.write(f"Model is loading. Please wait for {estimated_time} seconds...")
        time.sleep(estimated_time)
        output = query({"inputs": input_text})  # Query the API again

    print(output)

    # Process and display the response
    if 'error' not in output:
        entity_list_html = generate_entity_list(output)
        st.subheader("Entities found")
        st.markdown(entity_list_html, unsafe_allow_html=True)
    else:
        st.subheader("No entities found.")