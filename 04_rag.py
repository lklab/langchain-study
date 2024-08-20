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

# STEP 1. Indexing: Load
import bs4
from langchain_community.document_loaders import WebBaseLoader

# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

# print(len(docs[0].page_content))
# print(docs[0].page_content[:500])

# STEP 2. Indexing: Split
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# print(len(all_splits))
# print(len(all_splits[0].page_content))
# print(all_splits[10].metadata)

# STEP 3. Indexing: Store
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# STEP 4. Retrieval and Generation: Retrieve
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
retrieved_docs = retriever.invoke("What are the approaches to Task Decomposition?")

# print(len(retrieved_docs))
# print(retrieved_docs[0].page_content)

# STEP 5. Retrieval and Generation: Generate
from langchain import hub

prompt = hub.pull("rlm/rag-prompt")

example_messages = prompt.invoke(
    {"context": "filler context", "question": "filler question"}
).to_messages()

# print(example_messages)
# print()
# print(example_messages[0].content)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

for chunk in rag_chain.stream("What is Task Decomposition?"):
    print(chunk, end="", flush=True)

# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate

# system_prompt = (
#     "You are an assistant for question-answering tasks. "
#     "Use the following pieces of retrieved context to answer "
#     "the question. If you don't know the answer, say that you "
#     "don't know. Use three sentences maximum and keep the "
#     "answer concise."
#     "\n\n"
#     "{context}"
# )

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )

# question_answer_chain = create_stuff_documents_chain(model, prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# response = rag_chain.invoke({"input": "What is Task Decomposition?"})
# print(response["answer"])
# print()

# for document in response["context"]:
#     print(document)
#     print()
