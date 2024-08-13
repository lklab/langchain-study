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
model = ChatOpenAI(
    model="gpt-3.5-turbo",
)

# setup message history
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

def promptTemplates() :
    from langchain_core.messages import HumanMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    # setup prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    chain = prompt | model

    # setup MessageHistory
    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="messages",
    )
    config = {"configurable": {"session_id": "abc5"}}

    # send to model
    response = with_message_history.invoke(
        {"messages": [HumanMessage(content="hi! I'm todd")], "language": "Korean"},
        config=config,
    )
    print(response.content)

    response = with_message_history.invoke(
        {"messages": [HumanMessage(content="whats my name?")], "language": "Korean"},
        config=config,
    )
    print(response.content)

def managingHistory() :
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import HumanMessage, AIMessage, trim_messages

    messages = [
        HumanMessage(content="hi! I'm bob"),
        AIMessage(content="hi!"),
        HumanMessage(content="I like vanilla ice cream"),
        AIMessage(content="nice"),
        HumanMessage(content="whats 2 + 2"),
        AIMessage(content="4"),
        HumanMessage(content="thanks"),
        AIMessage(content="no problem!"),
        HumanMessage(content="having fun?"),
        AIMessage(content="yes!"),
    ]

    trimmer = trim_messages(
        max_tokens=65,
        strategy="last",
        token_counter=model,
        include_system=True,
        allow_partial=False,
        start_on="human",
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    from operator import itemgetter
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda

    chain = (
        RunnablePassthrough.assign(messages=itemgetter("messages") | RunnableLambda(lambda x: messages + x) | trimmer)
        | prompt
        | model
    )

    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="messages",
    )
    config = {"configurable": {"session_id": "abc20"}}

    response = with_message_history.invoke(
        {
            "messages": [HumanMessage(content="what math problem did i ask")],
            "language": "English",
        },
        config=config,
    )
    print(response.content)

    response = with_message_history.invoke(
        {
            "messages": [HumanMessage(content="whats my name?")],
            "language": "English",
        },
        config=config,
    )
    print(response.content)

    response = with_message_history.stream(
        {
            "messages": [HumanMessage(content="tell me a joke")],
            "language": "English",
        },
        config=config,
    )

    for r in response :
        print(r.content, end="|")

# simpleExample()
# messageHistory()
# promptTemplates()
managingHistory()
