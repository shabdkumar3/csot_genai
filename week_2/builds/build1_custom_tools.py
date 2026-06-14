"""
Build 1: Custom Tool Call Parser
=================================
Before modern SDKs handled tool calls natively, developers used custom text formats
that the model was prompted to emit. This build has you implement that pattern from
scratch: prompt the model to emit tool calls in a structured format, parse them, run
the corresponding Python function, and feed the result back.

This is NOT the production way to do it (Build 2 is). But doing it manually first
makes the mechanics obvious. The SDK is doing exactly this, just more robustly.

The format we'll use:
    The model emits tool calls wrapped in <tool_call> tags, like:

        I need to read the file first.

        <tool_call>
        {"name": "read_file", "arguments": {"path": "notes.txt"}}
        </tool_call>

    Your code finds the tag, parses the JSON, runs the function, and injects
    the result back as a <tool_response> in the next message.

Tasks:
  1. Complete `parse_tool_call` to extract name + arguments from a model response
  2. Complete `dispatch` to route a tool call to the right Python function
  3. Complete `run_agent` to implement the back-and-forth loop

Tools to implement:
  - read_file(path: str) -> dict    reads a file from disk and returns its content
  - write_file(path: str, content: str) -> dict    writes content to a file on disk

Before running, create a file called `sample.txt` with some text in it.
"""

import os
import re
import sys
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# using groq (openai-compatible)
client = OpenAI(
    api_key=os.getenv("groq_api_key"),
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a helpful file assistant with access to the following tools:

- read_file(path: str): reads a file from disk and returns its content
- write_file(path: str, content: str): writes content to a file on disk

When you need to use a tool, emit EXACTLY this format and nothing else after it:

<tool_call>
{"name": "TOOL_NAME", "arguments": {"arg1": "value1"}}
</tool_call>

After you receive the tool result in a <tool_response> block, continue your response
normally. Do not emit a tool_call and prose in the same turn. Pick one or the other.
"""


def read_file(path: str) -> dict:
    """
    Read a file from disk and return its content.
    Return {"content": ..., "path": ...} on success.
    Return {"error": ...} if the file doesn't exist or can't be read.
    """
    # open and read
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {"content": f.read(), "path": path}
    except Exception as e:
        return {"error": str(e)}


def write_file(path: str, content: str) -> dict:
    """
    Write content to a file on disk.
    Return {"success": True, "path": ..., "bytes_written": ...} on success.
    Return {"error": ...} on failure.

    Hint: open(path, 'w') and then f.write(content).
    """
    # open and write
    try:
        with open(path, "w", encoding="utf-8") as f:
            n = f.write(content)
        return {"success": True, "path": path, "bytes_written": n}
    except Exception as e:
        return {"error": str(e)}


def parse_tool_call(response_text: str) -> dict | None:
    """
    Extract a tool call from the model's response text.

    Returns a dict {"name": str, "arguments": dict} if a <tool_call> block is found,
    or None if there is no tool call in the response.

    The format to parse:
        <tool_call>
        {"name": "...", "arguments": {...}}
        </tool_call>

    Hint: use re.search() with re.DOTALL to find the block, then json.loads() the body.
    """
    # find the block
    match = re.search(r"<tool_call>(.*?)</tool_call>", response_text, re.DOTALL)
    if not match:
        return None

    # parse the json inside
    try:
        return json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return None


def strip_tool_call(response_text: str) -> str:
    """
    Return the response text with any <tool_call>...</tool_call> block removed.
    Useful for printing the model's prose without the raw tag.
    """
    # cut the block out
    cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", response_text, flags=re.DOTALL)
    return cleaned.strip()


TOOL_REGISTRY = {
    "read_file": read_file,
    "write_file": write_file,
}

def dispatch(name: str, arguments: dict) -> str:
    """
    Look up the tool by name, call it with the given arguments, and return a
    JSON string of the result.

    If the tool is not found, return: {"error": "Unknown tool: <name>"}
    If the call raises an exception, return: {"error": "<exception message>"}

    Always return a string (json.dumps the result dict).
    """
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})

    # run it
    try:
        return json.dumps(fn(**arguments))
    except Exception as e:
        return json.dumps({"error": str(e)})


MAX_ITERATIONS = 6

def run_agent(user_message: str) -> str:
    """
    Run the tool-calling agent loop for a single user message.

    Steps:
      1. Build the initial messages list with SYSTEM_PROMPT + user message.
      2. Call the model.
      3. Parse the response for a <tool_call>.
      4. If found: run the tool, inject a <tool_response> block into messages, go to 2.
      5. If not found: return the model's text (the final answer).
      6. If MAX_ITERATIONS reached: return an error string.

    The <tool_response> you inject back should look like:
        <tool_response>
        {"content": "Hello, world!", "path": "sample.txt"}
        </tool_response>

    Wrap it in a user message so the model sees it as a continuation:
        {"role": "user", "content": "<tool_response>\\n...\\n</tool_response>"}

    Print a line to stderr each time a tool is called so you can follow the loop.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    for iteration in range(MAX_ITERATIONS):
        # ask the model
        response = client.chat.completions.create(model=MODEL, messages=messages)
        reply = response.choices[0].message.content or ""

        call = parse_tool_call(reply)

        # no tool -> final answer
        if call is None:
            return reply.strip()

        # keep the assistant turn, then run the tool
        messages.append({"role": "assistant", "content": reply})

        name = call.get("name")
        args = call.get("arguments", {})
        print(f"[tool] {name}({args})", file=sys.stderr)

        result = dispatch(name, args)

        # feed result back as a tool_response
        messages.append(
            {"role": "user", "content": f"<tool_response>\n{result}\n</tool_response>"}
        )

    return f"[Agent stopped after {MAX_ITERATIONS} iterations]"


if __name__ == "__main__":
    # Create a sample file for the agent to work with
    with open("sample.txt", "w") as f:
        f.write("IIT Delhi was established in 1961. It is one of the premier engineering institutions in India.\n")
        f.write("The campus spans 325 acres in Hauz Khas, New Delhi.\n")

    test_queries = [
        "Read sample.txt and summarise what it says.",
        "Read sample.txt and write a one-sentence version of its content to summary.txt.",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        result = run_agent(query)
        print(f"Answer: {result}")
