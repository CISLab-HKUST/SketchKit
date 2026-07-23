# main.py
import os
import shutil
import time
from typing import List
import json
from pathlib import Path
from openai import OpenAI
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# from tools import mcp  # Removed direct import to use subprocess

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

app = FastAPI()

PROJECT_ROOT = Path(os.getcwd()).resolve()
OUTPUT_DIR = (PROJECT_ROOT / "output").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = str(OUTPUT_DIR)
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")


def to_web_path(path_str):
    path_str = str(path_str)
    # Check if absolute path matches OUTPUT_DIR
    if path_str.startswith(OUTPUT_DIR):
        return path_str.replace(OUTPUT_DIR, "/output")
    # Also check if it's a relative path from cwd/output (tools might return relative)
    # But tools usually return absolute paths or paths relative to cwd.
    # If path starts with "output/", convert to "/output/"
    cwd = os.getcwd()
    abs_path = os.path.abspath(path_str)
    if abs_path.startswith(OUTPUT_DIR):
        return abs_path.replace(OUTPUT_DIR, "/output")

    return path_str


# 配置 LLM (这里以 OpenAI 格式为例，可以是 DeepSeek, LocalLLM 等)
# 使用 Qwen API (DashScope)
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY", "sk-1096ee32d1b04d88979e94005bf0038a"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

MODEL_NAME = "qwen-plus"

STORAGE_DIR = "./temp_storage"
os.makedirs(STORAGE_DIR, exist_ok=True)

server_script = Path(__file__).parent / "tools.py"

# We use uv to run the tools.py script
server_params = StdioServerParameters(
    command="uv", args=["run", str(server_script)], env=os.environ.copy()
)

async def run_conversation_stream(user_command, uploaded_files):
    file_context = ", ".join(uploaded_files)
    print(file_context)
    # Server parameters
    # Assuming tools.py is in the same directory as this file


    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Get tools from MCP
                mcp_tools = await session.list_tools()
                tools = []
                # Handle potentially different return types depending on SDK version
                tool_list = (
                    mcp_tools.tools if hasattr(mcp_tools, "tools") else mcp_tools
                )

                for t in tool_list:
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
                        "content": f"""You are a file-processing assistant. You do not have visual capabilities, but you can process files by calling tools.
                        The user has currently uploaded the following files: {file_context}.
                        These files are stored on the server; when calling tools, you only need to provide the file name. Use the complete file path to call the function.
                        Follow the user's instructions strictly when invoking tools.
                        Do not include any file paths or file names in your replies.""",
                    },
                    {"role": "user", "content": user_command},
                ]

                # 2. 开始多轮对话循环
                generated_artifacts = []
                MAX_TURNS = 5  # 防止无限循环

                yield f"data: {json.dumps({'type': 'status', 'content': 'Processing request...'})}\n\n"

                for turn in range(MAX_TURNS):
                    print(f"🔄 Turn {turn + 1}/{MAX_TURNS}...")

                    # Stream Thinking / Status
                    yield f"data: {json.dumps({'type': 'status', 'content': f'Thinking (Turn {turn + 1})...'})}\n\n"

                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                    )

                    response_message = response.choices[0].message
                    tool_calls = response_message.tool_calls
                    messages.append(response_message)

                    # 如果没有工具调用，说明 LLM 这是最终回复，直接结束循环
                    if not tool_calls:
                        # Yield final text reply
                        yield f"data: {json.dumps({'type': 'text', 'content': response_message.content})}\n\n"
                        # Yield combined result summary
                        final_data = {
                            "reply": response_message.content,
                            "generated_files": [
                                to_web_path(p) for p in generated_artifacts
                            ],
                        }
                        yield f"data: {json.dumps({'type': 'done', 'content': final_data})}\n\n"
                        return

                    # 3. 如果有工具调用，执行它们
                    for tool_call in tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)

                        print(
                            f"🤖 模型决定调用工具: {function_name} 参数: {function_args}"
                        )
                        yield f"data: {json.dumps({'type': 'status', 'content': f'Calling tool: {function_name}...'})}\n\n"

                        result_list = []
                        try:
                            # 4. 通过MCP Client执行工具
                            result = await session.call_tool(
                                function_name, arguments=function_args
                            )

                            if hasattr(result, "content") and result.content:
                                for content in result.content:
                                    if content.type == "text":
                                        result_list.append(content.text)
                                    else:
                                        result_list.append(str(content))
                            else:
                                # Fallback or empty content
                                result_list.append(str(result))

                        except Exception as e:
                            tool_results = (
                                f"Error executing tool {function_name}: {str(e)}"
                            )
                            print(f"❌ 工具执行错误: {tool_results}")
                            result_list.append(tool_results)

                        generated_artifacts.extend(result_list)
                        result_str = str(result_list)

                        # Yield intermediate results !
                        for res_path in result_list:
                            web_path = to_web_path(res_path)
                            if web_path.startswith("/") or web_path.startswith("http"):
                                yield f"data: {json.dumps({'type': 'result', 'path': web_path, 'tool': function_name})}\n\n"

                        print(f"🔧 工具执行结果: {result_str}")

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": function_name,
                                "content": result_str,
                            }
                        )

                # 如果超过最大轮数
                yield f"data: {json.dumps({'type': 'text', 'content': 'Conversation limit reached.'})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'content': {}})}\n\n"

    except Exception as e:
        print(f"Fatal error in conversation: {e}")
        yield f"data: {json.dumps({'type': 'text', 'content': f'Server error: {str(e)}'})}\n\n"


@app.post("/process")
async def process_command(
    command: str = Form(...), files: List[UploadFile] = File(default=[])
):
    """
    接收用户上传的文件和指令，处理后返回结果 (Streaming)
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

    # Use StreamingResponse
    return StreamingResponse(
        run_conversation_stream(command, uploaded_filenames),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
