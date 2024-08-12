import json
import os

# load API key
with open('apikeys.json') as f:
    keys = json.load(f)

os.environ["OPENAI_API_KEY"] = keys['OPENAI_API_KEY']
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = keys['LANGCHAIN_API_KEY']

# setup model
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-3.5-turbo")

def simpleExample() :
    # setup messages
    from langchain_core.messages import HumanMessage
    from langchain_core.messages import AIMessage

    result = model.invoke(
        [
            HumanMessage(content="Hi! I'm Bob"),
            AIMessage(content="Hello Bob! How can I assist you today?"),
            HumanMessage(content="What's my name?"),
        ]
    )
    print(result.content)

def messageHistory() :
    from langchain_core.messages import HumanMessage
    from langchain_core.chat_history import (
        BaseChatMessageHistory,
        InMemoryChatMessageHistory,
    )
    from langchain_core.runnables.history import RunnableWithMessageHistory

    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    with_message_history = RunnableWithMessageHistory(model, get_session_history)

    # session abc2: 1
    config = {"configurable": {"session_id": "abc2"}}
    response = with_message_history.invoke(
        [HumanMessage(content="Hi! I'm Bob")],
        config=config,
    )
    print("[abc2] " + response.content)

    # session abc2: 2
    response = with_message_history.invoke(
        [HumanMessage(content="What's my name?")],
        config=config,
    )
    print("[abc2] " + response.content)

    # session abc3: 1
    config = {"configurable": {"session_id": "abc3"}}
    response = with_message_history.invoke(
        [HumanMessage(content="What's my name?")],
        config=config,
    )
    print("[abc3] " + response.content)

    # session abc2: 3
    config = {"configurable": {"session_id": "abc2"}}
    response = with_message_history.invoke(
        [HumanMessage(content="What's my name?")],
        config=config,
    )
    print("[abc2] " + response.content)

# simpleExample()
messageHistory()
