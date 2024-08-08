import json
import os

# load API key
with open('apikeys.json') as f:
    keys = json.load(f)

os.environ["OPENAI_API_KEY"] = keys['OPENAI_API_KEY']
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = keys['LANGCHAIN_API_KEY']

# initialize model
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4")

# send messages
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="Translate the following from English into Italian"),
    HumanMessage(content="hi!"),
]

aiMessage = model.invoke(messages)
print(aiMessage)

# parse output
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
result = parser.invoke(aiMessage)
print(result)
