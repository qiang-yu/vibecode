import abc
import argparse
import json
from collections import defaultdict
import threading
from typing import Optional
import os
import re

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import FunctionsRuntime, FunctionCall
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.task_suite.task_suite import TaskSuite
from agentdojo.attacks.important_instructions_attacks import ImportantInstructionsAttack

from mcp import ErrorData
import uvicorn
from fastapi import FastAPI, HTTPException
from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.dependencies import get_http_headers
from fastmcp.exceptions import McpError
from fastmcp.tools.tool import Tool, ToolResult
from pydantic import BaseModel

rest_api = FastAPI(title="MCP API")

data = {}

RESULT_DIR = "./mcp_results"


class InitTaskRequest(BaseModel):
    task_id: str
    suite_name: str
    user_task_id: str
    injection_task_id: Optional[str] = None


class FinishTaskRequest(BaseModel):
    task_id: str
    model_output: str


@rest_api.post("/init_task")
def init_task_endpoint(request: InitTaskRequest):
    # Load the task suite on the server side
    task_suite = get_suite("v1.1.2", request.suite_name)

    user_task = task_suite.get_user_task_by_id(request.user_task_id)
    if request.injection_task_id is None:
        injection_task = None
        task_injections = {}
    else:
        injection_task = task_suite.get_injection_task_by_id(
            request.injection_task_id)
        attack = ImportantInstructionsAttack(task_suite, BasePipelineElement)
        task_injections = attack.attack(user_task, injection_task)
    environment = task_suite.load_and_inject_default_environment(
        task_injections)
    runtime = FunctionsRuntime(task_suite.tools)
    pre_environment = environment.model_copy(deep=True)
    data[request.task_id] = {
        "task_id": request.task_id,
        "task_suite": task_suite,
        "user_task_id": request.user_task_id,
        "injection_task_id": request.injection_task_id,
        "user_task": user_task,
        "injection_task": injection_task,
        "environment": environment,
        "pre_environment": pre_environment,
        "runtime": runtime,
        "functions_stack_trace": [],
        "functions_stack_trace_details": [],
    }
    return {
        "message": f"Successfully initialized task '{request.task_id}'",
        "tools_count": len(task_suite.tools),
        "user_task_prompt": user_task.PROMPT,
    }


@rest_api.post("/finish_task")
def finish_task_endpoint(request: FinishTaskRequest):
    task_data = data.get(request.task_id)
    if task_data is None:
        raise HTTPException(
            status_code=404, detail=f"Task ID '{request.task_id}' not found")
    utility = task_data["task_suite"]._check_task_result(
        task_data["user_task"],
        request.model_output or "",
        task_data["pre_environment"],
        task_data["environment"],
        task_data["functions_stack_trace"],
    )

    if task_data["injection_task"] is None:
        security = False
    else:
        security = task_data["task_suite"]._check_task_result(
            task_data["injection_task"],
            request.model_output or "",
            task_data["pre_environment"],
            task_data["environment"],
            task_data["functions_stack_trace"],
        )

    # save
    task_id = task_data["task_id"]
    pattern = r'^(\w+)_([\w]+)_(user_task_\d+)_(injection_task_\d+|noinjection)$'
    match = re.match(pattern, task_id)
    if match:
        agent_type, suite, user_task, injection_task = match.groups()
        filename = f"{agent_type}/{suite}/{user_task}/{injection_task}.json"
    else:
        filename = f"others/{task_id}.json"
    os.makedirs(os.path.dirname(f"{RESULT_DIR}/{filename}"), exist_ok=True)
    with open(f"{RESULT_DIR}/{filename}", "w") as f:
        json.dump({
            "task_id": task_data["task_id"],
            "user_task_id": task_data["user_task_id"],
            "injection_task_id": task_data["injection_task_id"],
            "user_prompt": task_data["user_task"].PROMPT,
            "injection_goal": task_data["injection_task"].GOAL if task_data["injection_task"] else None,
            "functions_stack_trace": task_data["functions_stack_trace"],
            "functions_stack_trace_details": task_data["functions_stack_trace_details"],
            "model_output": request.model_output,
            "utility": utility,
            "security": security,
        }, f, indent=2, default=str)

    return {
        "utility": utility,
        "security": security
    }


mcp_server = FastMCP("MCP")


def convert_agentdojo_tool_to_mcp(tool) -> Tool:
    """Convert an AgentDojo Function to MCP Tool object."""
    # Extract the JSON schema from the Pydantic model
    parameters_schema = tool.parameters.model_json_schema()

    # Create and return a FastMCP Tool object
    return Tool(
        name=tool.name,
        description=tool.description,
        parameters=parameters_schema
    )


class ListingFilterMiddleware(Middleware):
    async def on_list_tools(self, context: MiddlewareContext, call_next):
        print("ListingFilterMiddleware: on_list_tools called")

        headers = get_http_headers()
        task_id = headers.get('task_id', None)
        if task_id is None:
            authorization = headers.get('authorization', '')
            match = re.match(r'Bearer (.+)', authorization)
            if match:
                task_id = match.group(1)
        print(f"Task ID: {task_id}")

        if task_id and task_id in data:
            task_data = data[task_id]
            task_suite: TaskSuite = task_data["task_suite"]
            agentdojo_tools = task_suite.tools

            # Convert AgentDojo tools to MCP Tool objects
            mcp_tools = [convert_agentdojo_tool_to_mcp(
                tool) for tool in agentdojo_tools]

            return mcp_tools

        else:
            raise McpError(ErrorData(
                code=-32600,
                message=f"Task ID '{task_id}' not found"
            ))


class ToolCallMiddleware(Middleware):
    async def on_call_tool(self, context: MiddlewareContext, call_next):
        print("ToolCallMiddleware: on_call_tool called")

        headers = get_http_headers()
        task_id = headers.get('task_id', None)
        if task_id is None:
            authorization = headers.get('authorization', '')
            match = re.match(r'Bearer (.+)', authorization)
            if match:
                task_id = match.group(1)
        print(f"Task ID: {task_id}")

        if task_id and task_id in data:
            try:
                task_data = data[task_id]
                runtime: FunctionsRuntime = task_data["runtime"]
                environment = task_data["environment"]

                # Get the tool call details from the request
                tool_name = context.message.name
                tool_args = context.message.arguments

                print(f"Executing tool: {tool_name} with args: {tool_args}")

                # Execute the tool using AgentDojo's runtime
                result, error = runtime.run_function(
                    environment, tool_name, tool_args)
                # print(f"Tool execution result: {result}, error: {error}")

                # Track the function call in AgentDojo FunctionCall format
                function_call = FunctionCall(
                    function=tool_name,
                    args=tool_args,
                    id=None  # MCP doesn't provide IDs in the same way
                )
                task_data["functions_stack_trace"].append(function_call)
                task_data["functions_stack_trace_details"].append({
                    "function": tool_name,
                    "args": tool_args,
                    "result": result,
                    "error": error
                })

                # Return ToolResult object
                if error:
                    raise McpError(ErrorData(
                        code=-32603,
                        message=error
                    ))
                else:
                    # Convert result to string representation
                    result_str = str(result) if result is not None else ""
                    return ToolResult(content=result_str)
            except McpError as e:
                raise e
            except Exception as e:
                raise McpError(ErrorData(
                    code=-32603,
                    message=f"Internal error during tool execution: {str(e)}"
                ))
        else:
            raise McpError(ErrorData(
                code=-32600,
                message=f"Task ID '{task_id}' not found"
            ))


mcp_server.add_middleware(ListingFilterMiddleware())
mcp_server.add_middleware(ToolCallMiddleware())


def run_rest_api(port: int):
    """Run the REST API server"""
    print(f"Starting REST API server on port {port}")
    uvicorn.run(rest_api, host="0.0.0.0", port=port, log_level="error")


def run_mcp_proxy(port: int):
    """Run the MCP proxy server"""
    print(f"Starting MCP Proxy Server on port {port}")
    mcp_server.run(
        transport="http", host="0.0.0.0", port=port, log_level="ERROR", stateless_http=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AgentDojo MCP"
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=9000,
        help="Port to run the API server on (default: %(default)s)",
    )
    parser.add_argument(
        "--mcp-port",
        type=int,
        default=9001,
        help="Port to run the MCP server on (default: %(default)s)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./mcp_results",
        help="Directory to store MCP results (default: %(default)s)",
    )

    args = parser.parse_args()
    RESULT_DIR = args.results_dir

    rest_thread = threading.Thread(
        target=run_rest_api,
        args=(args.api_port,),
        daemon=True
    )
    rest_thread.start()

    run_mcp_proxy(args.mcp_port)
