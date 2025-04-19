import os
import sys
import json
import logging
import asyncio
from typing import Optional, AsyncGenerator

from mcp import types as mcp_types
from mcp.server.lowlevel import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.loop_agent import LoopAgent
from google.adk.examples import VertexAiExampleStore
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools import FunctionTool
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.mcp_tool.conversion_utils import adk_to_mcp_tool_type
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types
from dotenv import load_dotenv
from utils.sqlite_database import get_db_instance

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mcp_server.log"),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("mcp_server")

LLM_MODEL = "gemini-2.0-flash-lite"
APP_NAME = "sql_query_mcp_server"

USER_ID = os.getenv("CLIENT_USER_ID", "mcp_server_user")
SESSION_ID = os.getenv("CLIENT_SESSION_ID", "mcp_server_session")

logger.debug(f"Using client user_id={USER_ID}, session_id={SESSION_ID}")

app = Server("sql-query-mcp-server")

logger.info("Creating SQL Query MCP Server...")


def list_tables(tool_context: ToolContext) -> dict:
    """Lists all tables in the database."""
    db = get_db_instance()
    if not db:
        return {"status": "error", "result": "Database not initialized."}
    try:
        tables = db.get_usable_table_names()
        tool_context.state["available_tables"] = tables
        return {"status": "success", "result": "Tables in database:\n" + "\n".join(tables)}
    except Exception as e:
        return {"status": "error", "result": f"Error listing tables: {e}"}


def get_table_schema(table_name: str, tool_context: ToolContext) -> dict:
    """Gets the schema for specified table(s)."""
    db = get_db_instance()
    if not db:
        return {"status": "error", "result": "Database not initialized."}
    try:
        schemas = []
        for table in table_name.split(","):
            info = db.get_table_info_no_throw(table.strip())
            schemas.append(f"Schema for {table.strip()}:\n{info}")
        result = "\n\n".join(schemas)
        tool_context.state["table_schemas"] = result
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "result": f"Error getting schema: {e}"}


def execute_query(query: str, tool_context: ToolContext) -> dict:
    """Executes SQL query and returns results."""
    db = get_db_instance()
    if not db:
        return {"status": "error", "result": "Database not initialized."}
    attempts = tool_context.state.get("query_attempts", [])
    attempts.append(query)
    tool_context.state.update({
        "query_attempts": attempts,
        "current_query": query,
        "query_success": False,
        "query_error": None,
        "query_result": None
    })
    try:
        result = db.run_no_throw(
            query) or "Query executed successfully but returned no results."
        tool_context.state.update({
            "query_result": result,
            "query_success": True,
            "last_successful_query": query
        })
        return {"status": "success", "result": result}
    except Exception as e:
        error_msg = f"Query execution error: {e}"
        tool_context.state.update({
            "query_error": error_msg,
            "query_success": False,
            "query_result": None
        })
        return {"status": "error", "result": error_msg}


class LoopTerminationChecker(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        query_success = ctx.session.state.get("query_success", False)
        iterations = ctx.session.state.get("iterations", 0)
        max_iterations = 3

        ctx.session.state["iterations"] = iterations + 1

        should_terminate = query_success or iterations >= max_iterations

        if should_terminate:
            reason = "query_succeeded" if query_success else "max_iterations_reached"
            ctx.session.state["termination_reason"] = reason

            content = types.Content(
                role="assistant",
                parts=[types.Part(text=f"Loop terminating. Reason: {reason}")]
            )
            yield Event(
                content=content,
                author=self.name,
                actions=EventActions(escalate=True)
            )
        else:
            content = types.Content(
                role="assistant",
                parts=[types.Part(
                    text=f"Continuing. Iteration: {iterations+1}")]
            )
            yield Event(
                content=content,
                author=self.name,
                actions=EventActions(escalate=False)
            )


def inject_examples_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest
) -> Optional[LlmResponse]:
    example_provider = None

    vertex_example_store = os.getenv("EXAMPLE_STORE")
    if vertex_example_store:
        try:
            example_provider = VertexAiExampleStore(
                examples_store_name=vertex_example_store)
        except Exception as e:
            logger.info(f"[Callback] Error initializing example provider: {e}")

    if not example_provider:
        logger.info(
            "[Callback] No example provider available. Skipping example injection.")
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
        logger.info(
            "[Callback] No user query found in request. Skipping example injection.")
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
        logger.info(
            f"[Callback] Injected {len(formatted_examples)} examples into the prompt")

    return None


sql_query_agent = LlmAgent(
    name="SQLQueryAgent",
    model=LLM_MODEL,
    instruction="""You are a technical SQL query generator and executor. Your primary job is to:

    1. List available tables using the list_tables tool
    2. Get schema information for relevant tables using the get_table_schema tool
    3. Generate syntactically correct SQLite queries
    4. Execute queries using the execute_query tool
    
    When writing queries:
    - Use proper SQLite syntax
    - Create appropriate joins when needed
    - Limit results to 5 rows unless requested otherwise
    - Avoid selecting all columns (*) when specific columns would suffice
    - Use proper filtering and sorting (ORDER BY) as needed
    
    For query refinement (when iterations > 0):
    - Check the previous failed query in 'current_query' state
    - Read the error message in 'query_error' state
    - Review past attempts in 'query_attempts' state
    - Make appropriate corrections to fix the issues
    
    IMPORTANT: You MUST call execute_query with your generated SQL. DO NOT just write out a SQL query.
    
    Return only the raw results from the database when successful or the error message when failed.
    DO NOT add any extra explanations or formatting to the results.
    """,
    tools=[
        FunctionTool(func=list_tables),
        FunctionTool(func=get_table_schema),
        FunctionTool(func=execute_query)
    ],
    before_model_callback=inject_examples_callback,  # Add the callback here
    output_key="execution_result"
)

loop_termination_checker = LoopTerminationChecker(
    name="LoopTerminationChecker")

query_loop = LoopAgent(
    name="QueryLoopAgentTool",
    sub_agents=[sql_query_agent, loop_termination_checker],
    max_iterations=5
)

sql_generator_tool = AgentTool(agent=query_loop)

session_service = InMemorySessionService()


async def process_sql_query(query_text: str, client_user_id: str = None, client_session_id: str = None) -> dict:
    logger.info(f"MCP Server: Processing SQL query: {query_text}")

    user_id = client_user_id or USER_ID
    session_id = client_session_id or SESSION_ID

    logger.info(
        f"MCP Server: Using user_id={user_id}, session_id={session_id}")

    session = session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    if not session:
        session = session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)

    session.state["query_success"] = False
    session.state["iterations"] = 0
    session.state["query_attempts"] = []
    session.state["current_query"] = None
    session.state["query_error"] = None
    session.state["query_result"] = None
    session.state["execution_result"] = None

    runner = Runner(
        agent=query_loop,
        app_name=APP_NAME,
        session_service=session_service
    )

    content = types.Content(role='user', parts=[types.Part(text=query_text)])

    try:
        events = runner.run(
            user_id=user_id,
            session_id=session_id,
            new_message=content
        )

        for event in events:
            # Process events as needed
            pass

        updated_session = session_service.get_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id)

        result = {
            "query_success": updated_session.state.get("query_success", False),
            "query_result": updated_session.state.get("query_result", "No results available."),
            "executed_sql": updated_session.state.get("last_successful_query", "No successful query."),
            "error": updated_session.state.get("query_error")
        }

        return result
    except Exception as e:
        logger.info(f"MCP Server: Error during agent execution: {e}")
        return {
            "query_success": False,
            "error": f"Error during execution: {str(e)}",
            "query_result": None,
            "executed_sql": None
        }


@app.list_tools()
async def list_tools() -> list[mcp_types.Tool]:
    logger.info("MCP Server: Received list_tools request.")
    mcp_tool_schema = adk_to_mcp_tool_type(sql_generator_tool)
    logger.info(f"MCP Server: Advertising tool: {mcp_tool_schema}")
    return [mcp_tool_schema]


@app.call_tool()
async def call_tool(
    name: str, arguments: dict
) -> list[mcp_types.Tool]:
    if name == sql_generator_tool.name:
        try:
            query_text = arguments.get("request")
            if not query_text:
                raise ValueError("Missing required parameter 'query_text'")

            if "request" not in arguments:
                error_msg = "Missing required parameter 'query_text'"
                logger.info(f"MCP Server: {error_msg}")
                return [mcp_types.TextContent(type="text",
                                              text=json.dumps({"error": error_msg}))]

            client_user_id = arguments.get("client_user_id")
            client_session_id = arguments.get("client_session_id")

            result = await process_sql_query(
                query_text,
                client_user_id=client_user_id,
                client_session_id=client_session_id
            )

            response_text = json.dumps(result, indent=2)
            logger.info(f"MCP Server: Tool '{name}' executed successfully.")
            return [mcp_types.TextContent(type="text", text=response_text)]

        except Exception as e:
            logger.error(f"MCP Server: Error executing tool '{name}': {e}")
            error_text = json.dumps(
                {"error": f"Failed to execute tool '{name}': {str(e)}"})
            return [mcp_types.TextContent(type="text", text=error_text)]
    else:
        logger.error(f"MCP Server: Tool '{name}' not found.")
        error_text = json.dumps({"error": f"Tool '{name}' not implemented."})
        return [mcp_types.TextContent(type="text", text=error_text)]


async def run_server():
    """Runs the MCP server over standard input/output."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        logger.info("MCP Server starting handshake...")
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name=app.name,
                server_version="0.1.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
        logger.info("MCP Server run loop finished.")

if __name__ == '__main__':
    try:
        asyncio.run(run_server())
    except Exception as e:
        print(f"An error occurred: {e}")
