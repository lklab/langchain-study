import json
import os

# load API key
with open('apikeys.json') as f:
    keys = json.load(f)

os.environ["OPENAI_API_KEY"] = keys['OPENAI_API_KEY']
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = keys['LANGCHAIN_API_KEY']

# setup LLM
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
)

def construct_retriever() :
    import bs4
    from langchain_chroma import Chroma
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    return retriever

def run_by_chains() :
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate

    ### Construct a retriever ###
    retriever = construct_retriever()

    ### Answer question ###
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", system_prompt),
    #         ("human", "{input}"),
    #     ]
    # )

    # question_answer_chain = create_stuff_documents_chain(llm, prompt)
    # rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # response = rag_chain.invoke({"input": "What is Task Decomposition?"})
    # print(response["answer"])

    ### Contextualize question ###
    from langchain.chains import create_history_aware_retriever
    from langchain_core.prompts import MessagesPlaceholder

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # create a rag chain
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # from langchain_core.messages import AIMessage, HumanMessage

    # chat_history = []

    # question = "What is Task Decomposition?"
    # ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})
    # print(ai_msg_1["answer"])

    # chat_history.extend(
    #     [
    #         HumanMessage(content=question),
    #         AIMessage(content=ai_msg_1["answer"]),
    #     ]
    # )

    # second_question = "What are common ways of doing it?"
    # ai_msg_2 = rag_chain.invoke({"input": second_question, "chat_history": chat_history})

    # print(ai_msg_2["answer"])

    ### Statefully manage chat history ###
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_core.runnables.history import RunnableWithMessageHistory

    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # send queries
    response = conversational_rag_chain.invoke(
        {"input": "What is Task Decomposition?"},
        config={"configurable": {"session_id": "abc123"}}, # constructs a key "abc123" in `store`.
    )
    print(response["answer"])

    response = conversational_rag_chain.invoke(
        {"input": "What are common ways of doing it?"},
        config={"configurable": {"session_id": "abc123"}},
    )
    print(response["answer"])

def run_by_agents() :
    ### Construct a retriever ###
    retriever = construct_retriever()

    ### Build retriever tool ###
    from langchain.tools.retriever import create_retriever_tool

    tool = create_retriever_tool(
        retriever,
        "blog_post_retriever",
        "Searches and returns excerpts from the Autonomous Agents blog post.",
    )
    tools = [tool]

    # create a memory saver
    from langgraph.checkpoint.memory import MemorySaver # need to install langgraph: pip install -U langgraph
    memory = MemorySaver()

    # create a agent
    from langgraph.prebuilt import create_react_agent
    agent_executor = create_react_agent(llm, tools, checkpointer=memory)

    # from langchain_core.messages import HumanMessage
    # query = "What is Task Decomposition?"

    # for s in agent_executor.stream(
    #     {"messages": [HumanMessage(content=query)]},
    # ):
    #     print(s)
    #     print("----")

    # send queries
    from langchain_core.messages import HumanMessage

    config = {"configurable": {"thread_id": "abc123"}}
    queries = [
        "Hi! I'm bob",
        "What is Task Decomposition?",
        "What according to the blog post are common ways of doing it? redo the search",
        "What's my name?",
    ]

    for query in queries :
        print("query: " + query)
        print("----")
        for s in agent_executor.stream(
            {"messages": [HumanMessage(content=query)]}, config=config
        ):
            print(s)
            print("----")
        print()

# run_by_chains()
run_by_agents()
