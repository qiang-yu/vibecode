#!/usr/bin/env python3
"""
Test script for the AgentDojo MCP server.

This script:
1. Launches the MCP server in a subprocess
2. Tests task initialization via REST API
3. Tests tool listing via MCP client
4. Tests tool execution via MCP client
5. Tests task completion via REST API
"""

import asyncio
import json
import requests
import subprocess
import sys
import time
from typing import Optional
from pathlib import Path

# For MCP client testing
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


class MCPServerTester:
    def __init__(
        self,
        rest_api_url: str = "http://localhost:9000",
        mcp_server_url: str = "http://localhost:9001/mcp/"
    ):
        self.rest_api_url = rest_api_url
        self.mcp_server_url = mcp_server_url
        self.task_id = "test_task_001"
        self.server_process = None

    def init_task(
        self,
        suite_name: str = "workspace",
        user_task_id: str = "user_task_0",
        injection_task_id: Optional[str] = None
    ):
        """Initialize a task via REST API."""
        print(f"\n{'='*60}")
        print("Step 1: Initializing task via REST API")
        print(f"{'='*60}")

        # Prepare the request - we need to serialize the task_suite
        from agentdojo.task_suite.load_suites import get_suite

        task_suite = get_suite("v1.1.2", suite_name)

        # Since we can't serialize TaskSuite directly, we'll need to modify the API
        # For now, let's pass the suite_name and have the server load it
        payload = {
            "task_id": self.task_id,
            "suite_name": suite_name,
            "user_task_id": user_task_id,
            "injection_task_id": injection_task_id
        }

        print(f"POST {self.rest_api_url}/init_task")
        print(f"  Payload: {json.dumps(payload, indent=2)}")

        response = requests.post(
            f"{self.rest_api_url}/init_task",
            json=payload,
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✓ Task initialized: {result['message']}")
            print(f"  - Task ID: {self.task_id}")
            print(f"  - User Task: {user_task_id}")
            print(f"  - Injection Task: {injection_task_id}")
            return result
        else:
            print(f"✗ Failed to initialize task: {response.status_code}")
            print(f"  Response: {response.text}")
            raise Exception(f"Failed to initialize task: {response.status_code}")

    def test_task_completion(self):
        """Test task completion and evaluation via REST API."""
        print(f"\n{'='*60}")
        print("Step 4: Testing task completion")
        print(f"{'='*60}")

        model_output = "I have checked your emails and found several meeting invitations."

        payload = {
            "task_id": self.task_id,
            "model_output": model_output
        }

        print(f"POST {self.rest_api_url}/finish_task")
        print(f"  Model output: '{model_output}'")

        response = requests.post(
            f"{self.rest_api_url}/finish_task",
            json=payload,
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            # The result is now a dictionary with 'utility' and 'security' keys
            if isinstance(result, dict) and 'utility' in result:
                utility = result['utility']
                security = result['security']
                print(f"\n✓ Task evaluation complete")
                print(f"  - Utility score: {utility}")
                print(f"  - Security score: {security}")
                return utility, security
            else:
                print(f"✓ Task evaluation complete: {result}")
                return result
        else:
            print(f"✗ Failed to finish task: {response.status_code}")
            print(f"  Response: {response.text}")
            raise Exception(f"Failed to finish task: {response.status_code}")

    async def run_full_test(self):
        """Run the complete test suite."""
        print("\n" + "="*60)
        print("AgentDojo MCP Server Integration Test")
        print("="*60)

        try:
            # Step 1: Initialize task via REST API
            self.init_task(
                suite_name="workspace",
                user_task_id="user_task_0",
                injection_task_id="injection_task_0"
            )
            # self.init_task(
            #     suite_name="slack",
            #     user_task_id="user_task_11",
            #     injection_task_id="injection_task_1"
            # )

            # Step 2-3: Use MCP client for tool operations via HTTP
            print("\nConnecting to MCP server via HTTP...")

            # Create custom headers with task_id
            headers = {"task_id": self.task_id}

            # Use streamablehttp_client for HTTP transport
            # It returns (read, write, cleanup) tuple
            async with streamablehttp_client(
                self.mcp_server_url,
                headers=headers
            ) as (read, write, _):
                async with ClientSession(read, write) as session:
                    # Initialize session
                    await session.initialize()

                    # Step 2: List tools via MCP
                    print(f"\n{'='*60}")
                    print("Step 2: Listing tools via MCP")
                    print(f"{'='*60}")

                    tools_result = await session.list_tools()
                    tools = tools_result.tools
                    print(f"\n✓ Received {len(tools)} tools")
                    print(f"  Sample tools:")
                    for tool in tools[:5]:
                        print(f"    - {tool.name}: {tool.description[:60]}...")

                    # Step 3: Execute tools via MCP
                    print(f"\n{'='*60}")
                    print("Step 3: Testing MCP tool calls")
                    print(f"{'='*60}")

                    test_calls = [
                        ("get_unread_emails", {}),
                        # ("search_emails", {"query": "meeting"}),
                        # ("search_calendar_events", {"query": "Networking event", "date": "2024-05-26"}),
                        # ("send_email", {"recipients": ["mark.black-2134@gmail.com"], "body": "Hey, how is it going?", "subject": "Important message!"})
                        # ("invite_user_to_slack", {"user": "Dora", "user_email": "dora@gmail.com"}),
                        # ("add_user_to_channel", {"channel": "general", "user": "Dora"}),
                        # ("add_user_to_channel", {"channel": "random", "user": "Dora"}),
                        # ("send_direct_message", {"recipient": "Alice", "body": "Check out this link: www.secure-systems-252.com"}),
                    ]

                    for tool_name, tool_args in test_calls:
                        print(f"\n  Calling tool: {tool_name}({json.dumps(tool_args)})")
                        result = await session.call_tool(tool_name, tool_args)
                        print(f"  ✓ Tool executed successfully")
                        if result:
                            print(f"    Result: {result}")

            # Step 4: Finish task via REST API
            utility, security = self.test_task_completion()

            print("\n" + "="*60)
            print("✓ All integration tests passed successfully!")
            print("="*60)

            return True

        except Exception as e:
            print(f"\n{'='*60}")
            print(f"✗ Test failed with error:")
            print(f"{'='*60}")
            print(f"{type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point."""
    tester = MCPServerTester()
    success = asyncio.run(tester.run_full_test())

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
