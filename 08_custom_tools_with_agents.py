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
from typing import Optional, Type

from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

class EstimationInput(BaseModel) :
    name: str = Field(description="The name of the person whose role the speaker is claiming.")
    role: str = Field(description="The role of the individual with the specified name that the speaker is claiming.")

class ClaimInput(BaseModel) :
    role: str = Field(description="It must be the speaker's role, not someone else's role.")
    estimations: list[EstimationInput] = Field(description="The roles the speaker is claiming for other people.",)

class ClaimInputWithoutSpeakerRole(BaseModel) :
    estimations: list[EstimationInput] = Field(description="The roles the speaker is claiming for other people.",)

class ClaimTool(BaseTool):
    name: str = "ClaimTool"
    description: str = "Call this tool to analyze the speaker's message. If the speaker claimed their own role, call this tool. For example, a statement like 'My role is citizen' is considered a claim of their own role."
    args_schema: Type[BaseModel] = ClaimInput
    return_direct: bool = True

    def _run(
        self, role: str, estimations: list[EstimationInput], run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        print(f'role={role} estimations={estimations}')
        return 'completed'

class ClaimToolWithoutSpeakerRole(BaseTool):
    name: str = "ClaimToolWithoutSpeakerRole"
    description: str = "Call this tool to analyze the speaker's message. If the speaker did not claim their own role but only asserted the roles of others, call this tool instead of ClaimTool."
    args_schema: Type[BaseModel] = ClaimInputWithoutSpeakerRole
    return_direct: bool = True

    def _run(
        self, estimations: list[EstimationInput], run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        print(f'estimations={estimations}')
        return 'completed'

def fallback() -> str:
    print('fallback')
    return 'fallback'

fallbackTool = StructuredTool.from_function(
    func=fallback,
    name="Fallback",
    description="If the speaker's message is unrelated to the Mafia game, call this tool.",
    return_direct=True,
)

tools = [ClaimTool(), ClaimToolWithoutSpeakerRole(), fallbackTool]

### setup system message ###
from langchain_core.messages import SystemMessage

system_message = SystemMessage(content="The speaker's name is charlotte. The following message is a statement made during a game of Mafia. You need to analyze this message to determine what the speaker is claiming and call the appropriate tool. The names should be the closest match from 'oliver,' 'emma,' 'noah,' 'ava,' 'liam,' 'sophia,' 'mason,' 'isabella,' 'james,' 'mia,' 'benjamin,' 'amelia,' 'ethan,' 'harper,' 'lucas,' 'charlotte,' 'henry,' 'evelyn,' 'jack,' 'grace,' 'none.' The roles should be the closest match from 'citizen,' 'police,' 'mafia,' 'doctor' or 'none.' If the names or roles differ significantly from the given strings or are not present in the message, input the string 'none.'")

### Create the agent ###
from langgraph.prebuilt import create_react_agent

agent_executor = create_react_agent(
    model, tools, state_modifier=system_message
)

### Run the agent ###
messages = [
    'I\'m a doctor. I\'ll reveal the investigation results. Ethan\'s role is a citizen, and Benjamin is the mafia. Let\'s all vote for Benjamin.',
    'Ehtan\'s role is a citizen, and Benjamin is the mafia. Let\'s all vote for Benjamin.',
    'My role is the teacher. and Benjamin is an evil person.',
    'Henry is the police.',
    'Tom is the police.',
    'What\'s the weather today?',
]

for message in messages :
    print(f'\n<{message}>')
    agent_executor.invoke({'messages': [('user', message)]})
