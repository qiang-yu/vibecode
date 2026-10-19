#!/usr/bin/env python3
"""
Run AgentDojo tasks for a given suite.

This script:
1. Takes a suite name as command line input
2. Runs all tasks under that suite (user_task only and combinations with injection_task)
3. For each task: init -> run agent (placeholder) -> finish
"""

import argparse
import asyncio
import json
import os
import requests
import subprocess
import sys
from typing import Optional


# Suite definitions
suites = {
    "banking": {
        "user_task": [f"user_task_{i}" for i in range(16)],
        "injection_task": [f"injection_task_{i}" for i in range(9)]
    },
    "slack": {
        "user_task": [f"user_task_{i}" for i in range(21)],
        "injection_task": [f"injection_task_{i}" for i in range(1, 6)]
    },
    "travel": {
        "user_task": [f"user_task_{i}" for i in range(20)],
        "injection_task": [f"injection_task_{i}" for i in range(7)]
    },
    "workspace": {
        "user_task": [f"user_task_{i}" for i in range(40)],
        "injection_task": [f"injection_task_{i}" for i in range(6)]
    }
}


class TaskRunner:
    """Runner for AgentDojo tasks."""

    def __init__(
        self,
        rest_api_url: str = "http://127.0.0.1:9000",
        mcp_server_url: str = "http://127.0.0.1:9001/mcp/",
        agent_type: str = "openai"
    ):
        self.rest_api_url = rest_api_url
        self.mcp_server_url = mcp_server_url
        self.agent_type = agent_type

    def get_task_id(self, suite_name: str, user_task_id: str, injection_task_id: Optional[str] = None) -> str:
        """Generate a unique task ID."""
        prefix = f"{self.agent_type}"
        ENABLE_SECAGENT = os.getenv("ENABLE_SECAGENT", "False").lower() == "true"
        if ENABLE_SECAGENT:
            prefix += "_progent"
        if injection_task_id:
            return f"{prefix}_{suite_name}_{user_task_id}_{injection_task_id}"
        else:
            return f"{prefix}_{suite_name}_{user_task_id}_noinjection"

    def init_task(
        self,
        task_id: str,
        suite_name: str,
        user_task_id: str,
        injection_task_id: Optional[str] = None
    ) -> dict:
        """Initialize a task via REST API."""
        payload = {
            "task_id": task_id,
            "suite_name": suite_name,
            "user_task_id": user_task_id,
            "injection_task_id": injection_task_id
        }

        print(f"  Initializing task: {task_id}")
        print(f"    Suite: {suite_name}")
        print(f"    User Task: {user_task_id}")
        print(f"    Injection Task: {injection_task_id}")

        response = requests.post(
            f"{self.rest_api_url}/init_task",
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            print(f"  ✓ Task initialized successfully")
            print(f"    User Prompt: {result.get('user_task_prompt', 'N/A')[:100]}...")
            return result
        else:
            print(f"  ✗ Failed to initialize task: {response.status_code}")
            print(f"    Response: {response.text}")
            raise Exception(f"Failed to initialize task: {response.status_code}")

    def finish_task(
        self,
        task_id: str,
        model_output: str
    ) -> tuple:
        """Finish a task and get evaluation via REST API."""
        payload = {
            "task_id": task_id,
            "model_output": model_output
        }

        print(f"  Finishing task: {task_id}")

        response = requests.post(
            f"{self.rest_api_url}/finish_task",
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, dict) and 'utility' in result:
                utility = result['utility']
                security = result['security']
                print(f"  ✓ Task completed - Utility: {utility}, Security: {security}")
                return utility, security
            else:
                print(f"  ✓ Task completed: {result}")
                return result, None
        else:
            print(f"  ✗ Failed to finish task: {response.status_code}")
            print(f"    Response: {response.text}")
            raise Exception(f"Failed to finish task: {response.status_code}")

    async def run_agent(self, task_id: str, user_prompt: str):
        """
        Run the agent for a task.

        Executes the appropriate agent based on self.agent_type.
        Currently supports: 'openai' (openai-agent-sdk.py), 'langchain' (langchain-agent.py), 'openhands' (openhands-agent.py)
        """
        print(f"  Running {self.agent_type} agent for task: {task_id}")

        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))

        if self.agent_type == "openai":
            agent_script = os.path.join(script_dir, "openai-agent-sdk.py")
        elif self.agent_type == "langchain":
            agent_script = os.path.join(script_dir, "langchain-agent.py")
        elif self.agent_type == "openhands":
            agent_script = os.path.join(script_dir, "openhands-agent.py")
        elif self.agent_type == "autogen":
            agent_script = os.path.join(script_dir, "autogen-multi-agent.py")
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")

        # Check if the agent script exists
        if not os.path.exists(agent_script):
            raise FileNotFoundError(f"Agent script not found: {agent_script}")

        # Run the agent script
        cmd = [
            sys.executable,  # Use the same Python interpreter
            agent_script,
            "--mcp-server-url", self.mcp_server_url,
            "--task-id", task_id,
            "--user-input", user_prompt
        ]

        print(f"    Executing: {' '.join(cmd)}")

        try:
            # Run the agent and capture output
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            # Print stderr to screen
            if stderr:
                stderr_text = stderr.decode()
                print(f"    stderr output:\n{stderr_text}", file=sys.stderr)

            if process.returncode != 0:
                print(f"  ✗ Agent execution failed with return code {process.returncode}")
                print(f"    stderr: {stderr.decode()[:500]}")
                raise Exception(f"Agent execution failed: {stderr.decode()}")

            output = stdout.decode().strip()
            print(f"  ✓ Agent execution completed")
            print(f"    Output preview: {repr(output)[:500]}...")
            print(f"    Output length: {len(output)} chars")

            # # Extract the final output (last line or full output)
            # # The agent prints "Running: <prompt>" and then the result
            # lines = output.split('\n')
            # # Find the final output after "Running:" message
            # final_output = output
            # for i, line in enumerate(lines):
            #     if line.startswith("Running:"):
            #         # Everything after this line is the result
            #         final_output = '\n'.join(lines[i+1:]).strip()
            #         break

            return output

        except Exception as e:
            print(f"  ✗ Error executing agent: {e}")
            raise

    async def run_single_task(
        self,
        suite_name: str,
        user_task_id: str,
        injection_task_id: Optional[str] = None
    ):
        """Run a single task: init -> agent -> finish."""
        task_id = self.get_task_id(suite_name, user_task_id, injection_task_id)

        try:
            # Step 1: Initialize task
            init_result = self.init_task(task_id, suite_name, user_task_id, injection_task_id)
            user_prompt = init_result.get('user_task_prompt', '')

            if not user_prompt:
                raise ValueError("No user_task_prompt returned from init_task")

            # Step 2: Run agent
            model_output = await self.run_agent(task_id, user_prompt)

            # Step 3: Finish task
            utility, security = self.finish_task(task_id, model_output)

            return {
                "task_id": task_id,
                "suite_name": suite_name,
                "user_task_id": user_task_id,
                "injection_task_id": injection_task_id,
                "utility": utility,
                "security": security,
                "status": "success"
            }

        except Exception as e:
            print(f"  ✗ Error running task {task_id}: {e}")
            return {
                "task_id": task_id,
                "suite_name": suite_name,
                "user_task_id": user_task_id,
                "injection_task_id": injection_task_id,
                "status": "error",
                "error": str(e)
            }

    async def run_suite(self, suite_name: str):
        """Run all tasks in a suite."""
        if suite_name not in suites:
            print(f"Error: Unknown suite '{suite_name}'")
            print(f"Available suites: {', '.join(suites.keys())}")
            return []

        suite = suites[suite_name]
        results = []

        print(f"\n{'='*60}")
        print(f"Running suite: {suite_name}")
        print(f"{'='*60}")
        print(f"User tasks: {len(suite['user_task'])}")
        print(f"Injection tasks: {len(suite['injection_task'])}")
        print()

        total_tasks = len(suite['user_task']) * (1 + len(suite['injection_task']))
        current_task = 0

        # Run all tasks
        for user_task_id in suite['user_task']:
            # Run user task only (no injection)
            current_task += 1
            print(f"\n[{current_task}/{total_tasks}] Running: {user_task_id} (no injection)")
            print("-" * 60)
            result = await self.run_single_task(suite_name, user_task_id, None)
            results.append(result)

            # Run user task with each injection task
            for injection_task_id in suite['injection_task']:
                current_task += 1
                print(f"\n[{current_task}/{total_tasks}] Running: {user_task_id} + {injection_task_id}")
                print("-" * 60)
                result = await self.run_single_task(suite_name, user_task_id, injection_task_id)
                results.append(result)

        return results

    def print_summary(self, results: list):
        """Print summary of results."""
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")

        total = len(results)
        success = sum(1 for r in results if r['status'] == 'success')
        error = sum(1 for r in results if r['status'] == 'error')

        print(f"Total tasks: {total}")
        print(f"Successful: {success}")
        print(f"Errors: {error}")

        if success > 0:
            utilities = [r['utility'] for r in results if r['status'] == 'success' and r['utility'] is not None]
            securities = [r['security'] for r in results if r['status'] == 'success' and r['security'] is not None]

            if utilities:
                avg_utility = sum(utilities) / len(utilities)
                print(f"Average Utility: {avg_utility:.4f}")

            if securities:
                avg_security = sum(securities) / len(securities)
                print(f"Average Security: {avg_security:.4f}")

        print(f"{'='*60}\n")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run AgentDojo tasks for a given suite or single task"
    )
    parser.add_argument(
        "--suite",
        type=str,
        choices=list(suites.keys()),
        help="Name of the suite to run (required unless running single task)"
    )
    parser.add_argument(
        "--user-task",
        type=str,
        help="Specific user task to run (e.g., user_task_0). If provided, runs single task instead of full suite."
    )
    parser.add_argument(
        "--injection-task",
        type=str,
        help="Specific injection task to run (e.g., injection_task_0). Optional, used with --user-task."
    )
    parser.add_argument(
        "--rest-api-url",
        type=str,
        default="http://127.0.0.1:9000",
        help="URL of the REST API server (default: http://127.0.0.1:9000)"
    )
    parser.add_argument(
        "--mcp-server-url",
        type=str,
        default="http://127.0.0.1:9001/mcp/",
        help="URL of the MCP server (default: http://127.0.0.1:9001/mcp/)"
    )
    parser.add_argument(
        "--agent-type",
        type=str,
        default="openai",
        choices=["openai", "langchain", "openhands", "autogen"],  # Add more agent types here as they become available
        help="Type of agent to use (default: openai)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save results as JSON"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.suite:
        parser.error("--suite is required")

    if args.injection_task and not args.user_task:
        parser.error("--injection-task requires --user-task to be specified")

    # Create runner
    runner = TaskRunner(
        rest_api_url=args.rest_api_url,
        mcp_server_url=args.mcp_server_url,
        agent_type=args.agent_type
    )

    # Run single task or full suite
    if args.user_task:
        # Run single task
        print(f"\n{'='*60}")
        print(f"Running single task from suite: {args.suite}")
        print(f"{'='*60}")
        print(f"User task: {args.user_task}")
        print(f"Injection task: {args.injection_task or 'None'}")
        print()

        result = await runner.run_single_task(
            args.suite,
            args.user_task,
            args.injection_task
        )
        results = [result]
    else:
        # Run full suite
        results = await runner.run_suite(args.suite)

    # Print summary
    runner.print_summary(results)

    # Save results if output path provided
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")

    return 0 if all(r['status'] == 'success' for r in results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
