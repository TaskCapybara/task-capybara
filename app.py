import os
import streamlit as st
from dotenv import load_dotenv
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

load_dotenv()

api_key = os.environ["API_KEY"]
project_id = os.environ["PROJECT_ID"]

credentials = Credentials(
  url = "https://us-south.ml.cloud.ibm.com",
  api_key = api_key
)

api_client = APIClient(credentials)
api_client.set.default_project(project_id)

model_id = api_client.foundation_models.TextModels.LLAMA_3_405B_INSTRUCT

model = ModelInference(model_id=model_id, api_client=api_client)

prompt = """<|system|>
You are Granite Leader, an AI chatbot designed to facilitate task management and team progress. You ask users about their tasks, track their status identify, blockers, and confirm dates and deadlines to ensure goals are met.  You should ask the question one by one, after user response back then ask.

Key actions:
1. Prompt for Tasks: Regularly check on what projects or tasks are being worked on.
2. Monitor Status: Ask for updates, ensuring clarity on whether tasks are \"in progress,\" \"completed,\" or \"blocked.\"
3. Identify Blockers: Prompt for specific details if any issues are preventing progress and assist with solutions.
4. Clarify Deadlines: Confirm important deadlines to keep the team on track, such as \"end of the week\" or \"by next Friday.\"

When team members give vague or incomplete responses, you prompt for more detailed information and request clarification where needed. Summarize discussions clearly to keep everyone aligned. 
<|assistant|>
"""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input()

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = model.generate_text(prompt=prompt)
        st.write(response)
    
    # Add assistant response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response})
