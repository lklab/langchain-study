import json
import os

with open('apikeys.json') as f:
    keys = json.load(f)

os.environ["OPENAI_API_KEY"] = keys['OPENAI_API_KEY']

from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4")

question = "대한민국의 수도는 어디인가요?"
ret = model.invoke(question)
print(ret)
