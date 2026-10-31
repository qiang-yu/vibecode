"""Quick smoke test for the CaMeL defense against a local vLLM server.

Run it BEFORE a full benchmark, once per model:
    export LOCAL_LLM_URL=http://localhost:19000/v1
    export LOCAL_LLM_KEY=NOKEY   # optional
    python util_scripts/camel_vllm_smoke_test.py Qwen3Base

It checks the two LLM calls CaMeL makes:
  1. Quarantined LLM (pydantic_ai, structured output via tool_choice="required")
  2. Privileged LLM (plain chat, must return exactly ONE ```python code block)
"""

import os
import re
import sys

import openai
import pydantic_ai
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "agentdojo", "defenses", "camel", "src"))
from camel.quarantined_llm import make_openai_compatible_model, query_quarantined_llm  # noqa: E402

print("pydantic_ai version:", getattr(pydantic_ai, "__version__", "unknown"))

model_id = sys.argv[1] if len(sys.argv) > 1 else os.getenv("LOCAL_LLM_MODEL_ID", "Qwen3Base")
base_url = (os.getenv("LOCAL_LLM_URL") or f"http://localhost:{os.getenv('LOCAL_LLM_PORT', 8000)}/v1").rstrip("/")
api_key = os.getenv("LOCAL_LLM_KEY") or "EMPTY"
print(f"model={model_id}  base_url={base_url}")

# 1. Quarantined LLM
q_model = make_openai_compatible_model(model_id, base_url, api_key)


class Contact(BaseModel):
    name: str
    email: str


try:
    res = query_quarantined_llm(
        q_model, "Extract the contact from this text: 'Please reply to Alice Smith at alice@example.com.'", Contact
    )
    print("[Q-LLM] OK  ->", res.name, res.email)
except Exception as e:
    print("[Q-LLM] FAIL ->", type(e).__name__, e)

# 2. Privileged LLM
client = openai.OpenAI(api_key=api_key, base_url=base_url)
resp = client.chat.completions.create(
    model=model_id,
    temperature=0.0,
    max_tokens=int(os.getenv("MAX_INFERENCE_TOKENS", 4096)),
    messages=[
        {"role": "system", "content": "Write Python code to solve the user's task. Reply with exactly one ```python code block."},
        {"role": "user", "content": "Print the sum of the numbers 1 to 10."},
    ],
)
content = resp.choices[0].message.content or ""
pattern = r"```([a-zA-Z0-9_+\-#]*)\n(.*?)\n```"
raw_blocks = re.findall(pattern, content, re.DOTALL)
print(
    f"[P-LLM] raw output: finish_reason={resp.choices[0].finish_reason}  "
    f"code_blocks={len(raw_blocks)}  has_<think>={'<think>' in content}"
)
# same processing as CaMeL's privileged_llm.py: keep only text after the last </think>
camel_content = content.split("</think>")[-1].strip() if "</think>" in content else content
blocks = re.findall(pattern, camel_content, re.DOTALL)
print(f"[P-LLM] what CaMeL sees: code_blocks={len(blocks)}  has_<think>={'<think>' in camel_content}")
if "<think>" in content and "</think>" not in content:
    print("[P-LLM] WARN -> <think> not closed, output was probably truncated; increase MAX_INFERENCE_TOKENS")
if len(blocks) != 1:
    print("[P-LLM] FAIL -> CaMeL requires exactly one code block. Content head:\n", camel_content[:800])
else:
    print("[P-LLM] OK")
