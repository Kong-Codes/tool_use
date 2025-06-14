import asyncio
import json
import random

from autogen_core import CancellationToken
from autogen_core.models import AssistantMessage, FunctionExecutionResult, FunctionExecutionResultMessage, UserMessage
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient


async def get_weather(city: str) -> str:
    temp = random.randint(10, 30)
    return "The weather in " + city + " is " + str(temp) + " degrees Celsius."


# # Create a function tool.
weather_tool = FunctionTool(get_weather, description="Get the weather.")

# Create the OpenAI chat completion client. Using OPENAI_API_KEY from environment variable.
model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")


async def main():
    # Create a user message.
    user_message = UserMessage(content="What is the weather in sf?", source="user")

    # Run the chat completion with the stock_price_tool defined above.
    cancellation_token = CancellationToken()
    create_result = await model_client.create(
        messages=[user_message], tools=[weather_tool], cancellation_token=cancellation_token
    )

    assert isinstance(create_result.content, list)
    arguments = json.loads(create_result.content[0].arguments)  # type: ignore
    tool_result = await weather_tool.run_json(arguments, cancellation_token)
    tool_result_str = weather_tool.return_value_as_string(tool_result)
    assert isinstance(tool_result_str, str), "Tool result should be a string."

    # Create a function execution result
    exec_result = FunctionExecutionResult(
        call_id=create_result.content[0].id,  # type: ignore
        content=tool_result_str,
        is_error=False,
        name=weather_tool.name,
    )

    # Make another chat completion with the history and function execution result message.
    messages = [
        user_message,
        AssistantMessage(content=create_result.content, source="assistant"),  # assistant message with tool call
        FunctionExecutionResultMessage(content=[exec_result]),  # function execution result message
    ]
    create_result = await model_client.create(messages=messages, cancellation_token=cancellation_token)  # type: ignore
    print(create_result.content)
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
