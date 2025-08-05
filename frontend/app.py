import streamlit as st
from streamlit_chat import message
import requests

st.title("🧮 Math Routing Agent - ChatBot")

# Initialize session state
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = []

if 'bot_response' not in st.session_state:
    st.session_state['bot_response'] = []

# Function to call FastAPI /rag_combined/ endpoint
def get_answer_from_api(question):
    try:
        response = requests.post(
            "http://127.0.0.1:8000/rag_combined/",
            json={
                "query": question,
                "top_k": 1
            }
        )
        if response.status_code == 200:
            return response.json().get("answer", "⚠️ No answer found.")
        else:
            return f"⚠️ Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Input box
def get_text():
    return st.text_input("🔢 Ask a JEE Math question:", key="input")

user_input = get_text()

if user_input:
    answer = get_answer_from_api(user_input)
    st.session_state['user_input'].append(user_input)
    st.session_state['bot_response'].append(answer)

# Display history
if st.session_state['user_input']:
    for i in range(len(st.session_state['user_input']) - 1, -1, -1):
        message(st.session_state["user_input"][i], key=f"user_{i}", avatar_style="icons")
        message(st.session_state["bot_response"][i], key=f"bot_{i}", is_user=True, avatar_style="miniavs")
