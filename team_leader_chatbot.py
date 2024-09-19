import streamlit as st
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes

from components.model import AppModel
from components.team_leader_chat_utils import Prompt

# Initialize model and prompt
model_id = ModelTypes.GRANITE_13B_CHAT_V2
parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 900,
    "repetition_penalty": 1
}
assistant = AppModel(model_id, parameters)

def generate_chat_page():
    # Render chat history
    prompt = Prompt()
    generate_chat_history_view(prompt)

    user_input = st.chat_input()
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                assistant_response = assistant.generate_response(prompt.get_final_prompt(user_input))
            st.write(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

def generate_chat_history_view(prompt):
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant":
                prompt.add_assistant_response(message["content"])
            else:
                prompt.add_user_input(message["content"])

# Streamlit UI
st.title('TaskCapybara Chatbot')
st.header('For team leader')

# Initialize session states
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

generate_chat_page()