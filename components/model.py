import os
from dotenv import load_dotenv
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes

load_dotenv()

# Configure project and credentials
api_key = os.environ["API_KEY"]
project_id = os.environ["PROJECT_ID"]


class AppModel:
    def __init__(self, model_id, parameters) -> None:
        credentials = Credentials(
            url = "https://us-south.ml.cloud.ibm.com",
            api_key = api_key
        )
        api_client = APIClient(credentials)
        api_client.set.default_project(project_id)
        self.model = ModelInference(
            model_id=model_id,
            params=parameters,
            api_client=api_client
        )

    def generate_response(self, prompt):
        return self.model.generate_text(prompt)

    
