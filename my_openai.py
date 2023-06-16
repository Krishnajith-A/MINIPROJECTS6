import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key ="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"

response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "hello u r a chatbot"},
    {"role": "user", "content": "Hello!"}
  ]
)

print(response.choices[0].message['content'])




# openai.api_key = "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"
        #completion = openai.ChatCompletion.create(
           # model="gpt-3.5-turbo",
        #    messages=[
         #       {"role": "system", "content": "Your job is to read through a set of product reviews from customers of a product, identify problems with the product and generate suggestions as of how to improve that product. Enumerate your suggestions."},
          #      {"role": "user", "content": data.head}
           # ]
        #)

       # completion_1=completion.choices[0].message
