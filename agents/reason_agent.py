from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama

"""
Open Source Reasoning and Coding Agent
Download models from: https://ollama.com/library/deepseek-r1
Check what size fits in your gpu
Very slow ... but very powerful
"""

llm = ChatOllama(model="deepseek-r1:14b")

llm_agent = create_react_agent(
    llm,
    tools=[],
    state_modifier="Large Language Model for Reasoning and Coding. "
    "Ask first, when there are complex questions.",
)
