import streamlit as st
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes

from components.model import AppModel
from components.team_member_chat_utils import Prompt, FaissEmbedder

# Initialize model and prompt
model_id = ModelTypes.LLAMA_3_70B_INSTRUCT
parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 900,
    "repetition_penalty": 1
}
assistant = AppModel(model_id, parameters)

# Streamlit UI
st.title('TaskCapybara Chatbot')
st.header('For team member')

# Initialize session states
if "is_logged_in" not in st.session_state:
    st.session_state.is_logged_in = False
    st.session_state.username = None
if "end_chat" not in st.session_state:
    st.session_state.end_chat = False
if st.session_state.is_logged_in and "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            prompt = Prompt()
            assistant_response = assistant.generate_response(prompt.get_final_prompt(user_input="hi"))
    st.session_state.chat_history.append({"username": st.session_state.username,"role": "assistant", "content": assistant_response})
    st.rerun()

# Generate pages
def generate_login_page():
    st.title("Login")
    username = st.text_input("Enter your username:")
    if st.button("Login"):
        st.session_state.is_logged_in = True
        st.session_state.username = username
        st.rerun()

def generate_chat_page():
    st.subheader(f"Welcome back {st.session_state.username}")
    # Render chat history
    prompt = Prompt()
    generate_chat_history_view(prompt)

    count = 0
    if not st.session_state.end_chat and count < 4:
        count += 1
        user_input = st.chat_input()
        if user_input:
            with st.chat_message("user"):
                st.write(user_input)
            st.session_state.chat_history.append({"username": st.session_state.username, "role": "user", "content": user_input})
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    assistant_response = assistant.generate_response(prompt.get_final_prompt(user_input))
                    has_end_chat_tag, assistant_response = prompt.is_end_chat(assistant_response)
                    st.session_state.end_chat = has_end_chat_tag
                if assistant_response != "":
                    st.write(assistant_response)
            if assistant_response != "":
                st.session_state.chat_history.append({"username": st.session_state.username, "role": "assistant", "content": assistant_response})
    else:
        handle_chat_end()

def handle_chat_end():
    st.markdown("Chat ended.")
    faiss_embedder = FaissEmbedder()
    faiss_embedder.embed_chat_history(st.session_state.chat_history)

    # Testing
    user_input = st.chat_input()
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        results = faiss_embedder.search(user_input)
        print(results)
    

def generate_chat_history_view(prompt):
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant":
                prompt.add_assistant_response(message["content"])
            else:
                prompt.add_user_input(message["content"])

if not st.session_state.is_logged_in:
    generate_login_page()
else:
    generate_chat_page()