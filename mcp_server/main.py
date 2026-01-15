# main.py
import os
import shutil
import time
from typing import List, Optional
import json
from openai import OpenAI
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
import uvicorn
from tools import mcp  # 导入我们在上面定义的mcp实例

app = FastAPI()

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

# 配置 LLM (这里以 OpenAI 格式为例，可以是 DeepSeek, LocalLLM 等)
# 使用 Qwen API (DashScope)
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY", "sk-1096ee32d1b04d88979e94005bf0038a"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

MODEL_NAME = "qwen-plus"

STORAGE_DIR = "./temp_storage"
os.makedirs(STORAGE_DIR, exist_ok=True)


async def run_conversation(user_command, uploaded_files):
    file_context = ", ".join(uploaded_files)

    # Get tools from MCP (convert to OpenAI format)
    mcp_tools = await mcp.list_tools()
    tools = []
    for t in mcp_tools:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.inputSchema,
                },
            }
        )

    messages = [
        {
            "role": "system",
            "content": f"""
         你是一个文件处理助手。你没有视觉能力，但你可以通过调用工具来处理文件。
         当前用户已上传的文件有: {file_context}。
         这些文件存储在服务器上，调用工具时只需提供文件名即可。
         严格按照用户的指令调用工具。
         """,
        },
        {"role": "user", "content": user_command},
    ]

    # 2. 开始多轮对话循环
    generated_artifacts = []
    MAX_TURNS = 5  # 防止无限循环

    for turn in range(MAX_TURNS):
        print(f"🔄 Turn {turn + 1}/{MAX_TURNS}...")
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        messages.append(response_message)  # 把它的回复（可能包含工具调用或最终文本）加入历史

        # 如果没有工具调用，说明 LLM 这是最终回复，直接结束循环
        if not tool_calls:
            return {
                "content": response_message.content,
                "tool_results": generated_artifacts,
            }

        # 3. 如果有工具调用，执行它们
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"🤖 模型决定调用工具: {function_name} 参数: {function_args}")

            try:
                # 4. 通过MCP执行工具 (本地直接调用)
                tool_results = await mcp.call_tool(function_name, function_args)
            except Exception as e:
                tool_results = f"Error executing tool {function_name}: {str(e)}"
                print(f"❌ 工具执行错误: {tool_results}")
            
            # Extract content from TextContent objects or valid return structures
            result = []
            
            # Check for FastMCP (content, ctx) tuple return
            if isinstance(tool_results, tuple) and len(tool_results) == 2:
                 content, ctx = tool_results
                 # Prefer raw result from context if available
                 if isinstance(ctx, dict) and 'result' in ctx:
                     raw_result = ctx['result']
                     if isinstance(raw_result, list):
                         result.extend([str(item) for item in raw_result])
                     else:
                         result.append(str(raw_result))
                 elif isinstance(content, list):
                     for item in content:
                        if hasattr(item, 'text'):
                            result.append(item.text)
                        else:
                            result.append(str(item))
            elif isinstance(tool_results, list):
                for item in tool_results:
                    if hasattr(item, 'text'):
                        result.append(item.text)
                    else:
                        result.append(str(item))
            else:
                 result.append(str(tool_results))
            
            generated_artifacts.extend(result)
            result_str = str(result)

            print(f"🔧 工具执行结果: {result_str}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": result_str,
                }
            )
        
        # 循环继续，把工具结果给 LLM，让它决定下一步...

    # 如果超过最大轮数
    return {
        "content": "对话轮次过多，已强制停止。",
        "tool_results": generated_artifacts,
    }


@app.post("/process")
async def process_command(
    command: str = Form(...), files: List[UploadFile] = File(default=[])
):
    """
    接收用户上传的文件和指令，处理后返回结果
    """
    uploaded_filenames = []

    timestamp = int(time.time())
    # 保存上传的文件到 STORAGE_DIR
    for index, file in enumerate(files):
        if file.filename:
            # 获取扩展名
            ext = os.path.splitext(file.filename)[1]
            # 新文件名: 时间戳_序号 + 扩展名 (例如 1715678900_0.png)
            new_filename = f"{timestamp}_{index}{ext}"
            file_path = os.path.join(STORAGE_DIR, new_filename)

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            uploaded_filenames.append(file_path)
            print(f"✅ 文件已保存: {file_path} (原文件名: {file.filename})")

    try:
        result = await run_conversation(command, uploaded_filenames)
        generated_files = result["tool_results"]
        # Convert absolute paths to web-accessible paths
        # e.g. /home/.../output/file.png -> /output/file.png
        processed_files = []
        for file_path in generated_files:
            file_path_str = str(file_path)
            if file_path_str.startswith(OUTPUT_DIR):
            # Replace the local directory path with the mounted URL path
                url_path = file_path_str.replace(OUTPUT_DIR, "/output")
                processed_files.append(url_path)
            else:
                processed_files.append(file_path_str)
        generated_files = processed_files

        return {
            "reply": result["content"],
            "processed_files": uploaded_filenames,
            "generated_files": generated_files,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
