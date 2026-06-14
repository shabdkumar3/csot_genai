"""
Build 2: Tool Calling with the OpenAI SDK
==========================================
Build 1 had you implement the tool-call round-trip by hand using a custom text format.
This build does the same thing the production way: using the OpenAI SDK's native
`tools` parameter, `tool_calls` response field, and `"role": "tool"` messages.

The mechanics are identical. You're still parsing a tool name, running a function,
and sending the result back. The difference is that the SDK handles the encoding
and the model is trained to produce structured JSON tool calls rather than freeform XML.

Implement the same two tools as Build 1:
  - get_weather(city: str) -> dict
  - calculate(expression: str) -> dict

Then complete the agent loop and watch the pattern become clean.

Stretch goals (not required):
  - Add a third tool: get_time(timezone: str) -> dict
  - Handle multiple tool_calls in a single response (the model can call several at once)
  - Add a token counter that prints total tokens used after the loop ends
"""

import os
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


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Returns the current weather for a given city. "
                "Call this whenever the user asks about weather, temperature, or climate. "
                "Do not guess weather. Always call this tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g. 'Delhi' or 'San Francisco'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit. Default to celsius.",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluates a mathematical expression and returns the result. "
                "Use this for any arithmetic the user asks about. "
                "Pass the expression as a string, e.g. '1337 * 42 + 7'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A Python arithmetic expression, e.g. '100 / 4 + 3'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]


def get_weather(city: str, unit: str = "celsius") -> dict:
    """
    Return realistic-looking fake weather data for the city.
    In production this would call a weather API.

    Return a dict like:
        {"city": city, "temperature": 28, "unit": unit, "condition": "partly cloudy"}
    """
    # fake but stable per city
    conditions = ["sunny", "partly cloudy", "cloudy", "light rain"]
    temp = 20 + (len(city) % 15)  # just to vary it a bit
    if unit == "fahrenheit":
        temp = round(temp * 9 / 5 + 32)
    return {
        "city": city,
        "temperature": temp,
        "unit": unit,
        "condition": conditions[len(city) % len(conditions)],
    }


def calculate(expression: str) -> dict:
    """
    Safely evaluate a math expression.
    Use eval() with restricted globals so imports and builtins are blocked.
    Return {"result": value} or {"error": message}.
    """
    # no builtins -> no imports/funcs
    try:
        value = eval(expression, {"__builtins__": {}}, {})
        return {"result": value}
    except Exception as e:
        return {"error": str(e)}


TOOL_REGISTRY = {
    "get_weather": get_weather,
    "calculate": calculate,
}

def dispatch(tool_call) -> str:
    """
    Execute a single tool_call object from the API response.

    tool_call has:
        tool_call.function.name       (the tool name)
        tool_call.function.arguments  (a JSON string of arguments)

    Return a JSON string of the result dict.
    On unknown tool or exception, return a JSON error dict.

    Note: tool_call.function.arguments is a *string*, not a dict. Parse it first.
    """
    name = tool_call.function.name
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})

    # args come as a string
    try:
        args = json.loads(tool_call.function.arguments or "{}")
        return json.dumps(fn(**args))
    except Exception as e:
        return json.dumps({"error": str(e)})


MAX_ITERATIONS = 8

def run_agent(user_message: str) -> str:
    """
    Run the agent loop using native SDK tool calling.

    Steps:
      1. Append the user message to history.
      2. Call client.chat.completions.create() with tools=TOOLS.
      3. If response.choices[0].finish_reason == "tool_calls":
           a. Append the assistant message (it contains .tool_calls) to history.
           b. For each tool_call in message.tool_calls:
                - dispatch it
                - append a {"role": "tool", "tool_call_id": ..., "content": ...} message
           c. Go to 2.
      4. If finish_reason == "stop": return message.content.
      5. If MAX_ITERATIONS reached: return an error string.

    Print to stderr whenever a tool executes so you can follow the loop.

    Hint: the assistant message you append in step 3a must be the raw message object,
    not a dict. The SDK accepts both, but keep it consistent with what the API returned.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools when appropriate."},
        {"role": "user", "content": user_message},
    ]

    for _ in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
        )
        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # model wants tools
        if finish_reason == "tool_calls":
            messages.append(message)  # raw assistant msg with tool_calls
            for tc in message.tool_calls:
                result = dispatch(tc)
                print(f"[tool] {tc.function.name}({tc.function.arguments}) -> {result[:60]}", file=sys.stderr)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            continue

        # plain answer -> done
        return message.content

    return f"[Agent stopped after {MAX_ITERATIONS} iterations without a final answer]"


if __name__ == "__main__":
    test_queries = [
        "What's the weather in Tokyo?",
        "Calculate: (2**10) - 1",
        "Compare the weather in London and Delhi, and tell me what 451 * 3 is.",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        result = run_agent(query)
        print(f"\nFinal answer:\n{result}")
