import os
import streamlit as st
from dotenv import load_dotenv
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes

load_dotenv()

api_key = os.environ["API_KEY"]
project_id = os.environ["PROJECT_ID"]
credentials = Credentials(
  url = "https://us-south.ml.cloud.ibm.com",
  api_key = api_key
)
api_client = APIClient(credentials)
api_client.set.default_project(project_id)

# model_id = api_client.foundation_models.TextModels.LLAMA_3_405B_INSTRUCT
model_id = ModelTypes.LLAMA_3_70B_INSTRUCT
parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 900,
    "repetition_penalty": 1
}
model = ModelInference(model_id=model_id, params=parameters, api_client=api_client)

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

After gathering all the information, summarize the information in the following format:
1. Current Tasks:
2. Tasks statuses:
3. Blockers:
4. Deadlines:

You should only end the conversation with the tag <EndChat> after user agrees to end the conversation.
The tag <EndChat> should be the suffix of your response, and no more text after that.
"""
current_prompt = initial_prompt

st.title('TaskCapybara Chatbot')

if "end_chat" not in st.session_state:
    st.session_state.end_chat = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

current_prompt = initial_prompt
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            current_prompt += f"<|start_header_id|>{message["role"]}<|end_header_id|>{message["content"]}"
        else:
            current_prompt += f"<|eot_id|><|start_header_id|>{message["role"]}<|end_header_id|>{message["content"]}<|eot_id|>"

if not st.session_state.end_chat:
    user_input = st.chat_input()
    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        formatted_user_input = f"<|begin_of_text|><|eot_id|><|start_header_id|>user<|end_header_id|>{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                model_response = model.generate_text(prompt=f"{current_prompt}{formatted_user_input}")
                if model_response.endswith("<EndChat>"):
                    st.session_state.end_chat = True
                    model_response = model_response[:-9]
            if model_response != "":
                st.write(model_response)
        if model_response != "":
            st.session_state.chat_history.append({"role": "assistant", "content": model_response})
else:
    st.markdown("Chat ended.")    
