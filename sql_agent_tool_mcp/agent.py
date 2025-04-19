import os
import asyncio
from typing import Optional

import vertexai
import mcp.server.stdio

from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from google.genai import types
from google.adk.examples import VertexAiExampleStore

from dotenv import load_dotenv


load_dotenv()

APP_NAME = "sql_agent_app"
USER_ID = "user_01"
SESSION_ID = "session_01"
LLM_MODEL = "gemini-2.0-flash-lite"
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
VERTEX_EXAMPLE_STORE = os.getenv("EXAMPLE_STORE")
MCP_SERVER_SCRIPT = "sql_query_server/sql_query_mcp_server.py"


def initialize_vertex_ai():
    """Initialize Vertex AI with the project and location."""
    if GOOGLE_CLOUD_PROJECT:
        vertexai.init(project=GOOGLE_CLOUD_PROJECT, location="us-central1")
    else:
        print("Warning: GOOGLE_CLOUD_PROJECT environment variable not set")


def inject_examples_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest
) -> Optional[LlmResponse]:
    example_provider = VertexAiExampleStore(
        examples_store_name=VERTEX_EXAMPLE_STORE)

    if not example_provider:
        print("[Callback] No example provider available. Skipping example injection.")
        return None

    user_query = ""
    if llm_request.contents:
        for content in reversed(llm_request.contents):
            if content.role == 'user' and content.parts and content.parts[0].text:
                text = content.parts[0].text
                if not text.startswith("For context:"):
                    user_query = text
                    break

    if not user_query:
        print("[Callback] No user query found in request. Skipping example injection.")
        return None

    examples = example_provider.get_examples(user_query)

    formatted_examples = []
    for example in examples:
        user_query = example.input.parts[0].text
        assistant_response = example.output[0].parts[0].text
        formatted_examples.append(
            f"Question: {user_query}\n{assistant_response}\n")

    examples_text = "\n".join(formatted_examples)
    original_instruction = llm_request.config.system_instruction

    instruction_text = ""
    if isinstance(original_instruction, types.Content) and original_instruction.parts:
        instruction_text = original_instruction.parts[0].text or ""
    elif isinstance(original_instruction, str):
        instruction_text = original_instruction

    if formatted_examples:
        examples_section = f"\n\nRelevant examples to learn from:\n{examples_text}"
        modified_text = instruction_text + examples_section

        if isinstance(original_instruction, types.Content):
            original_instruction.parts[0].text = modified_text
        else:
            original_instruction = types.Content(
                role="system",
                parts=[types.Part(text=modified_text)]
            )

        llm_request.config.system_instruction = original_instruction

    return None


async def get_mcp_tools_async():
    """Connect to the SQL Query MCP Server and get tools."""
    # Create the MCP toolset from the server
    tools, exit_stack = await MCPToolset.from_server(
        connection_params=StdioServerParameters(
            command='python',
            args=[MCP_SERVER_SCRIPT],  # Path to the MCP server script
            env={
                "CLIENT_USER_ID": USER_ID,
                "CLIENT_SESSION_ID": SESSION_ID
            }
        )
    )
    for tool in tools:
        print(f"Tool name: {tool.name}")

    return tools, exit_stack


async def get_root_agent_async():
    """Create the root agent with MCP tools."""
    tools, exit_stack = await get_mcp_tools_async()

    root_agent = LlmAgent(
        name="SQLResultsInterpreterAgent",
        model=LLM_MODEL,
        instruction="""You are a helpful database assistant who communicates with users in natural language.
    
    Your role is to:
    1. Take the user's natural language question about the database
    2. Use the SQL query tool which will handle all the technical SQL work by passing the user's question as the 'query_text' parameter
    3. Interpret the results and present them in a clear, conversational way
    
    When presenting results to the user:
    - Provide a natural language interpretation of what the data shows
    - Highlight the most important insights from the results
    - Format the data in a readable way if there's a lot of information
    - If the query failed, explain what went wrong in non-technical terms
    
    Always maintain a helpful, conversational tone and focus on answering the user's original question completely.
    DO NOT show the SQL query to the user unless they specifically ask for it.
    """,
        tools=tools
    )
    return root_agent, exit_stack


async def query_database_async(question: str):
    """Process a natural language query about the database using async approach."""
    initialize_vertex_ai()
    print(f"\n>>> Question: {question}")

    session_service = InMemorySessionService()

    session = session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    if not session:
        session = session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)

    session.state = {
        "query_success": False,
        "iterations": 0,
        "query_attempts": [],
        "original_question": question
    }

    root_agent, exit_stack = await get_root_agent_async()

    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service
    )

    content = types.Content(role='user', parts=[types.Part(text=question)])

    final_agent_response = "No final response event generated."
    try:
        events_async = runner.run_async(
            user_id=USER_ID,
            session_id=SESSION_ID,
            new_message=content
        )

        async for event in events_async:
            print(f"Event received: {type(event).__name__}")

            if hasattr(event, 'content') and event.content is not None:
                print(
                    f"  Content role: {getattr(event.content, 'role', 'unknown')}")

                if hasattr(event.content, 'parts') and event.content.parts:
                    for i, part in enumerate(event.content.parts):
                        print(f"  Part {i}: type={type(part).__name__}")

                        if hasattr(part, 'function_call') and part.function_call is not None:
                            print(
                                f"    Tool Call: {part.function_call.name if hasattr(part.function_call, 'name') else 'unknown'}")
                            print(
                                f"    Args: {part.function_call.args if hasattr(part.function_call, 'args') else 'none'}")

                        if hasattr(part, 'function_response') and part.function_response is not None:
                            print(
                                f"    Tool Response: {part.function_response.name if hasattr(part.function_response, 'name') else 'unknown'}")
                            print(
                                f"    Response: {getattr(part.function_response, 'response', 'none')}")

                        if hasattr(part, 'text') and part.text:
                            text_preview = part.text[:100] + \
                                ("..." if len(part.text) > 100 else "")
                            print(f"    Text: {text_preview}")

            if event.is_final_response():
                if hasattr(event, 'content') and event.content and hasattr(event.content, 'parts') and event.content.parts:
                    final_agent_response = event.content.parts[0].text
                    print(f"Final response: {final_agent_response[:100]}...")
                    break
                else:
                    print("Final response event has no content or parts")

    except Exception as e:
        print(f"!!! An unexpected error occurred during agent run: {e}")
        final_agent_response = f"An error occurred: {e}"
    finally:
        print("Closing MCP server connection...")
        await exit_stack.aclose()
        print("Cleanup complete.")

    print(f"<<< Answer: {final_agent_response}")
    return final_agent_response


def query_database(question: str):
    """Synchronous wrapper for query_database_async."""
    return asyncio.run(query_database_async(question))


async def main_async():
    print("SQL Agent Test Runner (MCP Pattern)")
    questions = ["Which artist has the most tracks in the database?"]

    for question in questions:
        await query_database_async(question)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
