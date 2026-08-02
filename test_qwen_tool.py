###
# This script loads the Qwen3-8B model with an optional LoRA adapter
# and runs a two-turn tool-calling conversation. It prints the raw
# model output for each generation step.
#
# The ENABLE_THINKING flag controls whether Qwen3 runs in thinking mode
# (model may emit <think>...</think> blocks) or no-thinking mode.
###

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_PATH="/home/qiangyu/Models/Qwen/Qwen3-8B"
# LORA_PATH=None
LORA_PATH="/home/qiangyu/Models/FineTune/Qwen/cekl_train_20260731_ce2.5_kl0.02_0.01_1.0_nosecurity_3epochs_8B/checkpoint-900"

# Set to True to enable Qwen3 thinking mode, False for no-thinking mode.
ENABLE_THINKING = True

################################################################################
# load
################################################################################

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="cuda",
    trust_remote_code=True
).eval()

if LORA_PATH and os.path.isdir(LORA_PATH):
    print(f"Loading LoRA adapter from {LORA_PATH}...")
    model = PeftModel.from_pretrained(
        model,
        LORA_PATH,
    )
    model = model.merge_and_unload()
elif LORA_PATH:
    print(f"Warning: LoRA path does not exist: {LORA_PATH}")

################################################################################
# tool definition
################################################################################

tools = [
    {
        "type":"function",
        "function":{
            "name":"create_task",
            "description":"Create a task entry",
            "parameters":{
                "type":"object",
                "properties":{
                    "priority":{
                        "type":"integer",
                        "description":"Task priority"
                    },
                    "title":{
                        "type":"string",
                        "description":"Task title"
                    },
                    "urgent":{
                        "type":"boolean",
                        "description":"Urgent flag"
                    }
                },
                "required":[
                    "priority",
                    "title",
                    "urgent"
                ]
            }
        }
    }
]

################################################################################
# helper
################################################################################

def run_generation(messages, tools=None):

    raw_prompt = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING
    )

    print()
    print("="*80)
    print("RAW INPUT")
    print("="*80)
    print(raw_prompt)

    inputs = tokenizer(
        raw_prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs=model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False
        )

    full=tokenizer.decode(
        outputs[0],
        skip_special_tokens=False
    )

    generated=tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=False
    )

    print()
    print("="*80)
    print("RAW OUTPUT")
    print("="*80)
    print(generated)

    return generated


################################################################################
# round 1
################################################################################

messages = [
    {
        "role":"user",
        "content":
        "Please use the tool to create a task "
        "with priority 5, "
        "title 'Database migration', "
        "and urgent true."
    }
]

tool_call_raw = run_generation(
    messages,
    tools
)

################################################################################
# simulate tool execution
################################################################################

tool_result = {
    "succ": True,
    "message": "Database migration finish"
}

################################################################################
# round 2
################################################################################
#
# Important:
# Keep assistant raw output exactly as generated
# Then append tool response
#
################################################################################

messages.append(
    {
        "role":"assistant",
        "content":tool_call_raw
    }
)

messages.append(
    {
        "role":"tool",
        "name":"create_task",
        "content":json.dumps(tool_result)
    }
)

final_answer = run_generation(
    messages,
    tools
)

print()
print("="*80)
print("FINAL ASSISTANT RESPONSE")
print("="*80)
print(final_answer)
