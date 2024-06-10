from openai import OpenAI
import os
import numpy as np

client = OpenAI(
    base_url=os.getenv('OPENAI_API_BASE_URL') if 'OPENAI_API_BASE_URL' in os.environ else None,
    api_key=os.getenv('OPENAI_API_KEY'),
)

def get_embedding(text, model="text-embedding-3-small"):
   return client.embeddings.create(input = [text], model=model).data[0].embedding


text = "It is a test."

print(np.array(get_embedding(text)).shape)