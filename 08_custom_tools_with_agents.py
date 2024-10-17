import json
import os

### Load API keys ###
with open('apikeys.json') as f:
    keys = json.load(f)

os.environ["OPENAI_API_KEY"] = keys['OPENAI_API_KEY']
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = keys['LANGCHAIN_API_KEY']

### setup model ###
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-3.5-turbo",
)

### Define tools ###
from langchain_core.tools import tool
from langchain_core.tools import StructuredTool

@tool
def doSomething(a: int, b: int) -> int:
    """Do something two numbers."""
    print(f'doSomething {a}, {b}')
    return a + b + 1

def idontknow() -> str:
    print(f'idontknow')
    return 'I don\'t know!!'

idontknowTool = StructuredTool.from_function(
    func=idontknow,
    name="idontknow",
    description="Use this tool when any other available tool.",
    return_direct=True,
)

tools = [doSomething, idontknowTool]

### Create the agent ###
from langgraph.prebuilt import create_react_agent

agent_executor = create_react_agent(model, tools)

### Run the agent ###
from langchain_core.messages import HumanMessage, SystemMessage

response = agent_executor.invoke({"messages": [
    # SystemMessage(content="Read the following question and use the appropriate tool to answer it. If the question does not correspond to any available tool, simply respond with 'I don't know.'"),
    HumanMessage(content="Do something 2 and 5?"),
]})
for message in response["messages"] :
    print(repr(message))
    print('-----')

response = agent_executor.invoke({"messages": [
    # SystemMessage(content="Read the following question and use the appropriate tool to answer it. If the question does not correspond to any available tool, simply respond with 'I don't know.'"),
    HumanMessage(content="What is weather today?"),
]})
for message in response["messages"] :
    print(repr(message))
    print('-----')
