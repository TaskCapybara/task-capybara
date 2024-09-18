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

# prompt = """<|system|>
# You are AI chatbot designed to facilitate task management and team progress. You ask users about their tasks, track their status identify, blockers, and confirm dates and deadlines to ensure goals are met.  You should ask the question one by one, after user response back then ask.

# Key actions:
# 1. Prompt for Tasks: Regularly check on what projects or tasks are being worked on.
# 2. Monitor Status: Ask for updates, ensuring clarity on whether tasks are \"in progress,\" \"completed,\" or \"blocked.\"
# 3. Identify Blockers: Prompt for specific details if any issues are preventing progress and assist with solutions.
# 4. Clarify Deadlines: Confirm important deadlines to keep the team on track, such as \"end of the week\" or \"by next Friday.\"

# When team members give vague or incomplete responses, you prompt for more detailed information and request clarification where needed. Summarize discussions clearly to keep everyone aligned. 
# <|assistant|>
# """

initial_prompt = """<|start_header_id|>system<|end_header_id|>
You are a helpful, friendly, and professional team management assistant designed to gather daily progress updates from team members. 
Your goal is to ask the right questions to get detailed updates on task progress, blockers, and timelines. 
You should ask one question at a time. 
You should also maintain a conversational tone, encourage clarity, and offer help when needed. 
Always summarize the information clearly.
"""
current_prompt = ""

# """<|start_header_id|>system<|end_header_id|>

# You are a helpful, friendly, and professional team management assistant designed to gather daily progress updates from team members. Your goal is to ask the right questions to get detailed updates on task progress, blockers, and timelines. You should ask one question at a time. You should also maintain a conversational tone, encourage clarity, and offer help when needed. Always summarize the information clearly.
# <|eot_id|><|start_header_id|>user<|end_header_id|>

# hi<|eot_id|><|start_header_id|>assistant<|end_header_id|>

# Hello! I'm here to help gather your daily progress update. I'll ask you a series of questions to get a clear understanding of where you are with your tasks. Don't worry, it'll be quick and easy!

# To get started, can you tell me: What's the most  qimportant task you're working on today, and what's its current status?<|eot_id|><|start_header_id|>user<|end_header_id|>

# Today I will be focusing on refactoring our system's logs<|eot_id|><|start_header_id|>assistant<|end_header_id|>

st.title('Task Capybara chatbot')
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    current_prompt = initial_prompt

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        # if message["role"] == "assistant":
        #     current_prompt += f"<|start_header_id|>{message["role"]}<|end_header_id|>{message["content"]}"
        # else:
        #     current_prompt += f"<|eot_id|><|start_header_id|>{message["role"]}<|end_header_id|>{message["content"]}<|eot_id|>"

user_input = st.chat_input()

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    formatQuestion = f"<|begin_of_text|><|eot_id|><|start_header_id|>user<|end_header_id|>{user_input}<|eot_id|>"
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = model.generate_text(prompt=f"{current_prompt}{formatQuestion}")
        st.write(response)
    
    # Add assistant response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    current_prompt += f"<|eot_id|><|start_header_id|>user<|end_header_id|>{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{response}"""
