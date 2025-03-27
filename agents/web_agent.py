from typing import Literal

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from .agent_state import State

from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools import YouTubeSearchTool, ClickTool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import (
    create_sync_playwright_browser,  # A synchronous browser is available, though it isn't compatible with jupyter.\n",	  },
)
from langchain_community.utilities import WikipediaAPIWrapper

"""
Web Browser Agent
"""

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
ddg_tool = DuckDuckGoSearchResults(max_results=4, output_format="list")
arxiv_tool = ArxivQueryRun()
youtube_tool = YouTubeSearchTool()
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
# browser_tools = PlayWrightBrowserToolkit.from_browser(sync_browser=create_sync_playwright_browser()).get_tools()

research_agent = create_react_agent(
    llm,
    tools=[ddg_tool, arxiv_tool, youtube_tool, wiki_tool],  # + browser_tools,
    state_modifier="You are a webbrowser. You may use the DuckDuckGo search engine to search the web for important "
    "information, ArXiv for research paper, youtube and wikipedia. You may also use the browser to "
    "navigate to webpages.",
)
