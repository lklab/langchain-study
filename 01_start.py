import json
import os

# load API key
with open('apikeys.json') as f:
    keys = json.load(f)

os.environ["OPENAI_API_KEY"] = keys['OPENAI_API_KEY']
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = keys['LANGCHAIN_API_KEY']

def simpleTest() :
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

def lcel() :
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    template = '{country}의 수도는 어디인가요?'
    prompt = PromptTemplate.from_template(template)

    model = ChatOpenAI(
        model='gpt-3.5-turbo',
        max_tokens=64,
        temperature=0.1,
    )

    parser = StrOutputParser()

    chain = prompt | model | parser

    answer = chain.invoke({'country': '대한민국'})
    print(answer)

def runnable() :
    from operator import itemgetter

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda
    from langchain_openai import ChatOpenAI

    prompt = ChatPromptTemplate.from_template("{a} + {b} 는 무엇인가요?")
    model = ChatOpenAI(
        model='gpt-3.5-turbo',
        max_tokens=64,
        temperature=0.1,
    )

    chain = (
        {
            "a": itemgetter("word1") | RunnableLambda(lambda x: len(x)),
            "b": itemgetter("word2") | RunnableLambda(lambda x: len(x)),
        }
        | prompt
        | model
    )

    answer = chain.invoke({"word1": "broccoli", "word2": "world"})
    print(answer)

# simpleTest()
lcel()
# runnable()
