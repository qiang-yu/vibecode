import json
from transformers import AutoTokenizer

# 1. 加载你的 Qwen 模型分词器
# 注意：确保你的 transformers 库版本较新，以支持最新的 Qwen 模板
model_id = "/home/qiangyu/Models/Qwen/Qwen3-8B"  # Qwen3 架构或 Qwen2.5 均适用
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# 2. 你的原始数据（从 JSON 文件或数据库读取）
raw_data = {
    "id": "55ac2e75-716b-4626-acec-38b69ec47349",
    "conversations": [
        {
            "from": "human",
            "value": "Can you tell me about the movie with the ID 'tt0111161'?"
        },
        {
            "from": "gpt",
            "value": "<tool_call>\n{\"name\": \"get_movie_details\", \"arguments\": {\"movie_id\": \"tt0111161\"}}\n</tool_call>"
        },
        {
            "from": "tool",
            "value": "<tool_response>\n{\"name\": \"get_movie_details\", \"content\": {\"title\": \"The Shawshank Redemption\"...}}\n</tool_response>"
        },
        {
            "from": "gpt",
            "value": "The movie with the ID 'tt0111161' is \"The Shawshank Redemption\"..."
        }
    ],
    "tools": "[{\"type\": \"function\", \"function\": {\"name\": \"get_movie_details\", \"description\": \"Get details of a movie\", \"parameters\": {\"type\": \"object\", \"properties\": {\"movie_id\": {\"type\": \"string\", \"description\": \"The ID of the movie\"}}, \"required\": [\"movie_id\"]}}}, {\"type\": \"function\", \"function\": {\"name\": \"create_todo\", \"description\": \"Create a new todo item\", \"parameters\": {\"type\": \"object\", \"properties\": {\"task\": {\"type\": \"string\", \"description\": \"The task description\"}, \"due_date\": {\"type\": \"string\", \"format\": \"date\", \"description\": \"The due date of the task\"}, \"priority\": {\"type\": \"string\", \"enum\": [\"low\", \"medium\", \"high\"], \"description\": \"The priority level of the task\"}}, \"required\": [\"task\"]}}}]"
}

# 3. 角色映射字典 (将你的数据格式对齐到标准的 Hugging Face 格式)
role_mapping = {
    "system": "system",
    "human": "user",
    "gpt": "assistant",
    "tool": "tool"
}

# 4. 转换对话历史
messages = []
for msg in raw_data["conversations"]:
    mapped_role = role_mapping.get(msg["from"])
    if mapped_role:
        messages.append({
            "role": mapped_role,
            "content": msg["value"]
        })

# 5. 解析外部的 tools 字段（从 JSON 字符串转回 Python 列表对象）
# 因为 apply_chat_template 接收的是一个正规的 Python 列表/字典结构
tools_list = json.loads(raw_data["tools"])

# 6. 调用 apply_chat_template
# tokenize=False 表示只生成拼接好的纯文本（Prompt），便于查看或用于一些特定的训练框架
# add_generation_prompt=False 因为我们已经包含了完整的对话历史（包含了最后的 GPT 回复）
# 如果你是要在人问完后让大模型续写，则应该截断历史并设置 add_generation_prompt=True
prompt_text = tokenizer.apply_chat_template(
    conversation=messages,
    tools=tools_list,
    tokenize=False,
    add_generation_prompt=False 
)

print(prompt_text)
