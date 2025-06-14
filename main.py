import asyncio
import json
import os
from dataclasses import dataclass
from typing import List

from autogen_core import (
    AgentId,
    FunctionCall,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    message_handler,
)
from autogen_core import CancellationToken
from autogen_core.models import AssistantMessage, FunctionExecutionResult, FunctionExecutionResultMessage
from autogen_core.models import (
    ChatCompletionClient,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import FunctionTool, Tool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

@dataclass
class Message:
    content: str


class ToolUseAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, tool_schema: List[Tool]) -> None:
        super().__init__("An agent with tools")
        self._system_messages: List[LLMMessage] = [SystemMessage(content="""You are a helpful AI assistant that has access to tools. 
        When you need to find information that you don't know, you should use the search_web tool to look it up.
        Always use the search_web tool when asked about current or future information, especially about stock prices or other real-time data.""")]
        self._model_client = model_client
        self._tools = tool_schema

    @message_handler
    async def handle_user_message(self, message: Message, ctx: MessageContext) -> Message:
        # Create a session of messages.
        session: List[LLMMessage] = self._system_messages + [UserMessage(content=message.content, source="user")]

        # Run the chat completion with the tools.
        create_result = await self._model_client.create(
            messages=session,
            tools=self._tools,
            cancellation_token=ctx.cancellation_token,
        )

        # If there are no tool calls, return the result.
        if isinstance(create_result.content, str):
            return Message(content=create_result.content)
        assert isinstance(create_result.content, list) and all(
            isinstance(call, FunctionCall) for call in create_result.content
        )

        # Add the first model create result to the session.
        session.append(AssistantMessage(content=create_result.content, source="assistant"))

        # Execute the tool calls.
        results = await asyncio.gather(
            *[self._execute_tool_call(call, ctx.cancellation_token) for call in create_result.content]
        )

        # Add the function execution results to the session.
        session.append(FunctionExecutionResultMessage(content=results))

        # Run the chat completion again to reflect on the history and function execution results.
        create_result = await self._model_client.create(
            messages=session,
            cancellation_token=ctx.cancellation_token,
        )
        assert isinstance(create_result.content, str)

        # Return the result as a message.
        return Message(content=create_result.content)

    async def _execute_tool_call(
            self, call: FunctionCall, cancellation_token: CancellationToken
    ) -> FunctionExecutionResult:
        # Find the tool by name.
        tool = next((tool for tool in self._tools if tool.name == call.name), None)
        assert tool is not None

        # Run the tool and capture the result.
        try:
            arguments = json.loads(call.arguments)
            result = await tool.run_json(arguments, cancellation_token)
            return FunctionExecutionResult(
                call_id=call.id, content=tool.return_value_as_string(result), is_error=False, name=tool.name
            )
        except Exception as e:
            return FunctionExecutionResult(call_id=call.id, content=str(e), is_error=True, name=tool.name)


async def search_web(query: str) -> dict:
    """
    Search for the given query using the Tavily API.
    """
    # Perform search using Tavily API
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    results = tavily_client.search(query)
    return results


async def main():
    # Create the model client.
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
    # Create a runtime.
    runtime = SingleThreadedAgentRuntime()
    search_tool = FunctionTool(search_web, description="Get the recent data from the web")
    # Create the tools.
    tools: List[Tool] = [search_tool]
    # Register the agents.
    await ToolUseAgent.register(
        runtime,
        "tool_use_agent",
        lambda: ToolUseAgent(
            model_client=model_client,
            tool_schema=tools,
        ),
    )
    # Start processing messages.
    runtime.start()
    # Send a direct message to the tool agent.
    tool_use_agent = AgentId("tool_use_agent", "default")
    response = await runtime.send_message(Message("is iran at war?"), tool_use_agent)
    print(response.content)
    # Stop processing messages.
    await runtime.stop()
    await model_client.close()


if __name__ == "__main__":
    # Run the main function in an asyncio event loop.
    asyncio.run(main())
