import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class Prompt:
    def __init__(self) -> None:
        self.prompt = """<|start_header_id|>system<|end_header_id|>
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
    def add_chat_message(self, role, content):
        if role == "user":
            self.add_user_input(content)
        else:
            self.add_assistant_response(content)

    def add_user_input(self, user_input):
        self.prompt += f"<|eot_id|><|start_header_id|>user<|end_header_id|>{user_input}<|eot_id|>"

    def add_assistant_response(self, assistant_response):
        self.prompt += f"<|start_header_id|>assistant<|end_header_id|>{assistant_response}"

    def get_final_prompt(self, user_input):
        return f"{self.prompt}<|begin_of_text|><|eot_id|><|start_header_id|>user<|end_header_id|>{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    
    def is_end_chat(self, assistant_response):
        has_end_chat_tag = assistant_response.endswith("<EndChat>")
        if has_end_chat_tag:
            assistant_response = assistant_response[:-9]
        return has_end_chat_tag, assistant_response
    

class FaissEmbedder:
    DIMENSION = 384
    def __init__(self) -> None:
        self.index = faiss.IndexFlatL2(FaissEmbedder.DIMENSION)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def embed_chat_history(self, chat_history):
        with open("team_members_chat_histories.txt", "a") as file:
            for chat_message in chat_history:
                file.write(f"username: {chat_message["username"]}, role: {chat_message["role"]}, content: {chat_message["content"].replace("\n", " ")}\n")
        with open("team_members_chat_histories.txt", "r") as file:
            all_chat_history = file.read().splitlines()
        embeddings = self.embedder.encode(all_chat_history)
        embeddings = np.array(embeddings).astype(np.float32)
        self.index.add(embeddings)

    def save(self):
        faiss.write_index(self.index, 'faiss_index.index')
        