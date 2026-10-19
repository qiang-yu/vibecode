# AgentDojo MCP Server Test

## Overview

This test script validates the AgentDojo MCP server integration by:
1. Launching the MCP server with both REST API and MCP endpoints
2. Initializing a task via REST API
3. Listing available tools via MCP protocol
4. Executing tools via MCP protocol
5. Completing and evaluating the task via REST API

## Prerequisites

Make sure you have all dependencies installed:
```bash
pip install fastmcp mcp requests agentdojo
```

## Running the Test

Simply run the test script:

```bash
python test_mcp_server.py
```

The script will:
- Start the MCP server on ports 9080 (MCP) and 9081 (REST API)
- Run through all test steps
- Display detailed output for each step
- Automatically clean up and stop the server when done

## Test Steps

### Step 0: Server Startup
- Launches `mcp_server.py` in a subprocess
- Waits for the server to become ready
- Verifies the REST API is accessible

### Step 1: Task Initialization
- POST to `/init_task` with:
  - Suite name (e.g., "workspace")
  - User task ID (e.g., "user-task-0")
  - Optional injection task ID
- Server loads the task suite and environment
- Returns task ID and tool count

### Step 2: Tool Listing
- Sends MCP `tools/list` request with task_id in headers
- Server returns all available tools in MCP format
- Each tool includes name, description, and input schema

### Step 3: Tool Execution
- Sends MCP `tools/call` requests for:
  - `get_unread_emails` (no arguments)
  - `search_emails` (with query parameter)
- Server executes tools using AgentDojo runtime
- Tracks function calls for evaluation
- Returns tool results in MCP format

### Step 4: Task Completion
- POST to `/finish_task` with:
  - Task ID
  - Model output (final response)
- Server evaluates:
  - Utility (task completion)
  - Security (no injection executed)
- Returns evaluation scores

## Expected Output

```
============================================================
AgentDojo MCP Server Integration Test
============================================================

============================================================
Starting MCP Server
============================================================
Launching server: /path/to/mcp_server.py
  - REST API port: 9081
  - MCP port: 9080
Waiting for server to start... ✓ Server is ready!

============================================================
Step 1: Initializing task via REST API
============================================================
POST http://localhost:9081/init_task
  Payload: {...}
✓ Task initialized: Successfully initialized task 'test_task_001'
  - Task ID: test_task_001
  - User Task: user-task-0
  - Injection Task: None

============================================================
Step 2: Testing MCP tool listing
============================================================
Connecting to MCP server...
Sending MCP request: tools/list
✓ Received 17 tools
  Sample tools:
    - get_unread_emails: Returns all the unread emails in the inbox...
    - search_emails: Returns all emails that match the query in the body or subject...
    ...

============================================================
Step 3: Testing multiple MCP tool calls
============================================================

  Calling tool: get_unread_emails({})
  ✓ Tool executed successfully
    Result: [Email(id_=1, sender='alice@company.com', ...)]...

  Calling tool: search_emails({"query": "meeting"})
  ✓ Tool executed successfully
    Result: [Email(id_=3, ...)]...

✓ Completed 2 tool calls

============================================================
Step 4: Testing task completion
============================================================
POST http://localhost:9081/finish_task
  Model output: 'I have checked your emails and found several meeting invitations.'

✓ Task evaluation complete
  - Utility score: 1.0
  - Security score: True

============================================================
✓ All integration tests passed successfully!
============================================================

Stopping server...
✓ Server stopped
```

## Troubleshooting

### Port Already in Use
If you see "Address already in use" errors, the ports may be occupied:
```bash
# Check what's using the ports
lsof -i :9080
lsof -i :9081

# Kill existing processes if needed
kill -9 <PID>
```

### Server Fails to Start
- Check that all dependencies are installed
- Verify the AgentDojo data files are present
- Check the server logs in the subprocess output

### Test Failures
- The test will print detailed error messages
- Check the traceback for specific issues
- Verify the AgentDojo suite "workspace" is available

## Manual Testing

You can also start the server manually and test with curl:

```bash
# Start server
python mcp_server.py --port 9080 --api-port 9081

# Initialize task
curl -X POST http://localhost:9081/init_task \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "test_001",
    "suite_version": "v1.1.2",
    "suite_name": "workspace",
    "user_task_id": "user-task-0"
  }'

# List tools via MCP
curl -X POST http://localhost:9080 \
  -H "Content-Type: application/json" \
  -H "task_id: test_001" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list",
    "params": {}
  }'

# Call a tool via MCP
curl -X POST http://localhost:9080 \
  -H "Content-Type: application/json" \
  -H "task_id: test_001" \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
      "name": "get_unread_emails",
      "arguments": {}
    }
  }'

# Finish task
curl -X POST http://localhost:9081/finish_task \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "test_001",
    "model_output": "I checked your emails."
  }'
```
