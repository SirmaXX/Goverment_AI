from langchain_community.tools import DuckDuckGoSearchRun
import time

import ollama
from langchain_core.tools import tool
from PIL import Image
from pydantic import BaseModel, Field
from typing import Annotated

from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL


# Define the input schema for the tool
class FileInput(BaseModel):
    image_path: str = Field(description="should be a filepath")


repl = PythonREPL()


@tool("web_search", q=str, return_direct=True)
def web_search(q):
    """
    Performs a web search on DuckDuckGo based on the given query.

    Args:
        q (str): The word or phrase to search for.

    Returns:
        str: The first response from the search (returned by the langchain tool).

    Notes:
        Currently the function uses a fixed query (“Obama's first name?”).
        To make it dynamic, it should be changed to `search.invoke(q)`.
    """
    search = DuckDuckGoSearchRun()
    response = search.invoke(q)
    return response


@tool("calculator", math_ex=str, return_direct=True)
def calculator(math_exp):
    """
    Evaluates the given mathematical expression with the Python eval() function and returns the result.

    Args:
        math_exp (str): The mathematical expression to calculate (e.g. “2 + 3 * 4”).

    Returns:
        any: The result of the calculated mathematical expression.

    Warning:
        Since this function uses `eval()`, there may be a security risk with external input.
        Data from the user should not be run directly with eval.
    """
    return eval(math_exp)


# Define the tool
@tool("image-search-tool", args_schema=FileInput, return_direct=True)
def image_search(image_path: str) -> str:
    """Recognize images and generate a description."""
    response = ollama.chat(
        model="llama3.2-vision",  # Specify the vision model
        messages=[
            {
                "role": "user",  # The role of the message (user, assistant, etc.)
                "content": "What is in this image?",  # Your question about the image
                "images": [image_path],  # Path to the image
            }
        ],
    )
    # Access the 'message' key in the response dictionary
    return response["message"]["content"]


@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code and do math. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    print(f"Executing code: {code}")
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return result_str
