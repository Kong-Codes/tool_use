# AutoGen Tool-Using Agent Demo

This project demonstrates the implementation of AI agents using the AutoGen framework, showcasing how to create agents that can use tools to perform various tasks like web searches and weather queries.

## Features

- Asynchronous AI agent implementation using AutoGen
- Integration with OpenAI's GPT models
- Web search capabilities using Tavily API
- Tool-using functionality for real-time information retrieval
- Environment variable management for secure API key handling
- Type-safe implementation with error handling

## Project Structure

- `main.py`: Main implementation of the ToolUseAgent with web search capabilities
- `app.py`: Simple example demonstrating tool usage with a mock weather function

## Prerequisites

- Python 3.11 or higher
- OpenAI API key
- Tavily API key (for web search functionality)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## Usage

### Running the Main Agent

The main agent (`main.py`) demonstrates a tool-using agent that can perform web searches:

```bash
python main.py
```

This will start an agent that can:
- Process user queries
- Use web search tools to find information
- Provide responses based on real-time data

### Running the Weather Example

The weather example (`app.py`) shows a simpler implementation of tool usage:

```bash
python app.py
```

This demonstrates:
- Creating and using function tools
- Basic chat completion flow
- Function call handling
- Result processing

## Implementation Details

### ToolUseAgent

The `ToolUseAgent` class in `main.py` implements:
- Asynchronous message handling
- Tool execution capabilities
- Web search integration
- Error handling and type safety

### Weather Tool Example

The `app.py` file demonstrates:
- Basic tool creation
- Chat completion with tools
- Function execution
- Result processing

## Security Notes

- Never commit API keys or sensitive information
- Use environment variables for all API keys
- Keep your `.env` file in `.gitignore`

