import os
import json
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
model_details = model.get_details()
print(json.dumps( model_details, indent=2 ))
