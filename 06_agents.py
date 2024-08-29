import json
import os

### Load API keys ###
with open('apikeys.json') as f:
    keys = json.load(f)

os.environ["OPENAI_API_KEY"] = keys['OPENAI_API_KEY']
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = keys['LANGCHAIN_API_KEY']
os.environ["TAVILY_API_KEY"] = keys['TAVILY_API_KEY']

### Define tools ###
from langchain_community.tools.tavily_search import TavilySearchResults

search = TavilySearchResults(max_results=2)
# search_results = search.invoke("what is the weather in SF")
# print(search_results)

# If we want, we can create other tools.
# Once we have all the tools we want, we can put them in a list that we will reference later.
tools = [search]

### Using Language Models ###
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

model = ChatOpenAI(
    model="gpt-3.5-turbo",
)

# model_with_tools = model.bind_tools(tools)

# from langchain_core.messages import HumanMessage

# response = model_with_tools.invoke([HumanMessage(content="Hi!")])

# print(f"ContentString: {response.content}")
# print(f"ToolCalls: {response.tool_calls}")

# response = model_with_tools.invoke([HumanMessage(content="What's the weather in SF?")])

# print(f"ContentString: {response.content}")
# print(f"ToolCalls: {response.tool_calls}")

### Create the agent ###
from langgraph.prebuilt import create_react_agent

agent_executor = create_react_agent(model, tools)

### Run the agent ###
# response = agent_executor.invoke({"messages": [HumanMessage(content="hi!")]})
# print(response["messages"])

# response = agent_executor.invoke(
#     {"messages": [HumanMessage(content="whats the weather in sf?")]}
# )
# print(response["messages"])

### Streaming Messages ###
# for chunk in agent_executor.stream(
#     {"messages": [HumanMessage(content="whats the weather in sf?")]}
# ):
#     print(chunk)
#     print("----")

### Streaming tokens ###
# async def astream_agent(agent, query) :
#     async for event in agent.astream_events({"messages": [HumanMessage(content=query)]}, version="v1") :
#         kind = event["event"]
#         if kind == "on_chain_start":
#             if (
#                 event["name"] == "Agent"
#             ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
#                 print(
#                     f"Starting agent: {event['name']} with input: {event['data'].get('input')}"
#                 )
#         elif kind == "on_chain_end":
#             if (
#                 event["name"] == "Agent"
#             ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
#                 print()
#                 print("--")
#                 print(
#                     f"Done agent: {event['name']} with output: {event['data'].get('output')['output']}"
#                 )
#         if kind == "on_chat_model_stream":
#             content = event["data"]["chunk"].content
#             if content:
#                 # Empty content in the context of OpenAI means
#                 # that the model is asking for a tool to be invoked.
#                 # So we only print non-empty content
#                 print(content, end="|")
#         elif kind == "on_tool_start":
#             print("--")
#             print(
#                 f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}"
#             )
#         elif kind == "on_tool_end":
#             print(f"Done tool: {event['name']}")
#             print(f"Tool output was: {event['data'].get('output')}")
#             print("--")

# import asyncio
# asyncio.run(astream_agent(agent_executor, "whats the weather in sf?"))

### Adding in memory ###
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# # thread: abc123
# config = {"configurable": {"thread_id": "abc123"}}

# query = "hi im bob!"
# print(f"query: {query}")
# for chunk in agent_executor.stream(
#     {"messages": [HumanMessage(content=query)]}, config
# ):
#     print(chunk)
#     print("----")
# print()

# query = "whats my name?"
# print(f"query: {query}")
# for chunk in agent_executor.stream(
#     {"messages": [HumanMessage(content=query)]}, config
# ):
#     print(chunk)
#     print("----")
# print()

# # thread: xyz123
# config = {"configurable": {"thread_id": "xyz123"}}

# query = "whats my name?"
# print(f"query: {query}")
# for chunk in agent_executor.stream(
#     {"messages": [HumanMessage(content=query)]}, config
# ):
#     print(chunk)
#     print("----")
# print()
