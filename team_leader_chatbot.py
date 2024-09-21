import streamlit as st
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes

from components.model import AppModel
from components.team_leader_chat_utils import Prompt, FaissRetriever

# Initialize model and prompt
model_id = ModelTypes.GRANITE_13B_CHAT_V2
parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 900,
    "repetition_penalty": 1,
    "temperature": 0.2,
    "top_k": 3,
    "top_p": 1
}
assistant = AppModel(model_id, parameters)
retriever = FaissRetriever()

# Streamlit UI
logo_url = "logo.png"
st.set_page_config(page_title="TaskCapybara - Team Leader", page_icon=logo_url)
st.title('TaskCapybara - Team Leader')

def get_avatar(role):
    return logo_url if role == "assistant" else "😺"

# Initialize session states
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    with st.chat_message("assistant", avatar=get_avatar("assistant")):
        with st.spinner("Thinking..."):
            prompt = Prompt()
            prompt.with_retriever(retriever)
            question = """Please summarize all team members' task progress in the following format:
            Username:
            1. Current Tasks:
            2. Tasks statuses:
            3. Blockers:
            4. Deadlines:
            """
            assistant_response = assistant.generate_response(prompt.get_final_prompt(user_input=question))
    st.session_state.chat_history.append({"role": "assistant", "content": "Here is the update from the team members today: \n" + assistant_response})
    st.rerun()

# Generate pages
def generate_chat_page():
    # Render chat history
    prompt = Prompt()
    prompt.with_retriever(retriever)
    generate_chat_history_view(prompt)

    user_input = st.chat_input()
    if user_input:
        with st.chat_message("user", avatar=get_avatar("user")):
            st.write(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.chat_message("assistant", avatar=get_avatar("assistant")):
            with st.spinner("Thinking..."):
                assistant_response = assistant.generate_response(prompt.get_final_prompt(user_input))
            st.write(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

def generate_chat_history_view(prompt):
    for chat_message in st.session_state.chat_history:
            with st.chat_message(chat_message["role"], avatar=get_avatar(chat_message["role"])):
                st.write(chat_message["content"])
                prompt.add_chat_message(chat_message["role"], chat_message["content"])

generate_chat_page()