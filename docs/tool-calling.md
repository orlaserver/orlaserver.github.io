# Tool Calling

Tools let your LLM stages invoke Python functions during inference. The model decides when to call a tool based on the conversation, and Orla routes the tool call back to your code for execution. Tool calls never go through the Orla daemon. They run in your Python process.

## Tools with LangGraph

If you are using LangGraph, define tools with `@tool` from LangChain and bind them to the stage with `bind_tools`. This is the same pattern as any LangGraph application:

```python
from langchain_core.tools import tool
from pyorla import Stage

@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add a and b."""
    return a + b

tools = [multiply, add]

stage = Stage("calculator", backend)
stage.client = client
stage.set_max_tokens(512)

model_with_tools = stage.as_chat_model().bind_tools(tools)
```

`bind_tools` accepts LangChain `BaseTool` instances, plain dicts with `name`/`description`/`parameters`, and callable functions. It clones the stage internally so the original stage is not modified.

From here, use `model_with_tools` in your LangGraph nodes as you normally would. The tool node calls the tool functions directly in your process.

## Tools with the native Orla SDK

If you are using the Orla SDK directly instead of LangGraph, define tools with the `@orla_tool` decorator:

```python
from pyorla.tool_decorators import orla_tool

@orla_tool
def search(query: str, limit: int = 10) -> dict:
    """Search the knowledge base."""
    return {"hits": ["result1", "result2"]}
```

The decorator inspects the function signature, generates a JSON Schema from the type annotations, and wraps the function as a `Tool` that Orla can send to the backend. The first line of the docstring becomes the tool description.

Add the tool to a stage and execute:

```python
stage = Stage("search-agent", backend)
stage.client = client
stage.add_tool(search)
stage.set_max_tokens(512)

response = stage.execute("Find articles about KV cache optimization")
```

If the model responds with tool calls, parse and execute them:

```python
if response.tool_calls:
    results = stage.run_tool_calls_in_response(response)
    # results is a list of ToolResult, each with output_values or error
```

`run_tool_calls_in_response` parses each tool call from the response, looks up the matching tool by name on the stage, executes it with the arguments the model provided, and returns the results. You can then feed these results back as messages for the next inference call.

## Creating tools from dicts

If you have a tool schema but the execution logic is defined elsewhere, create a tool from a dict:

```python
from pyorla.tools import new_tool

def run_lookup(args: dict) -> dict:
    return {"status": "found", "data": args["id"]}

lookup = new_tool(
    name="lookup",
    description="Look up a record by ID",
    input_schema={
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Record ID"}
        },
        "required": ["id"],
    },
    run=run_lookup,
)

stage.add_tool(lookup)
```

## Converting LangChain tools

If you have existing LangChain tools and want to use them with the native SDK:

```python
from pyorla.langchain_tools import tool_from_langchain

orla_tool = tool_from_langchain(my_langchain_tool)
stage.add_tool(orla_tool)
```

This preserves the tool's name, description, schema, and execution logic.

## How tool calls flow

1. You add tools to a stage. When building the request, Orla converts each tool to the MCP wire format and includes them in the inference request.
2. The LLM backend receives the tools as part of the chat completion request and may respond with one or more tool calls instead of text.
3. Orla returns the raw tool calls in the response. The calls are not executed by the daemon.
4. Your code executes the tool calls locally using `stage.run_tool_calls_in_response()` or through LangGraph's tool node.
5. The results are appended to the message history and sent back for the next inference call.

This design keeps tool execution in your process, where it has access to your databases, APIs, and local state. The daemon only handles the LLM inference leg.
