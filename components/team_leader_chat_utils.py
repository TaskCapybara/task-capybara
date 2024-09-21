import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class Prompt:
    def __init__(self) -> None:
        self.prompt = """<|system|>
        You are Granite Chat, an AI language model developed by IBM. Your task is to answer questions 
        based on the given context, which is formatted to include username, role and the content 
        information based on the corresponding user's chat histories. Same username corresponds
        to a user, disregard of the roles.
    
        You should answer strictly based on the context as an assistant.
        If you don't know the answer to a question, just answer "Sorry, I don't have the information 
        to answer the question". Please do not make up responses. Please do not answer based on 
        chat histories of different users that are not related to the question.
        Ensure you reference the correct user's details.
        """
        self.retriever = None
    
    def with_retriever(self, retriever):
        self.retriever = retriever

    def add_chat_message(self, role, content):
        if role == "user":
            self.add_user_input(content)
        else:
            self.add_assistant_response(content)

    def add_user_input(self, user_input):
        self.prompt += f"<|user|>{user_input}"

    def add_assistant_response(self, assistant_response):
        self.prompt += f"<|assistant|>{assistant_response}"

    def get_final_prompt(self, user_input):
        retrieval_results = self.retriever.search(user_input)
        retrieval_prompt = "\n".join(retrieval_results)
        return f"{self.prompt}\nContext: \n{retrieval_prompt}\n<|user|>{user_input}<|assistant|>"
    

class FaissRetriever:
    def __init__(self) -> None:
        self.index = faiss.read_index('faiss_index.index')
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        with open("team_members_chat_histories.txt", "r") as file:
            self.chat_history = file.read().splitlines()

    def search(self, query, k=15):
        query_embedding = self.embedder.encode([query]).astype(np.float32)
        distances, indices = self.index.search(query_embedding, k)
        results = []
        print(indices[0])
        for idx in indices[0]:
            if idx >= 0 and idx < len(self.chat_history):
                results.append(self.chat_history[idx])
        return results