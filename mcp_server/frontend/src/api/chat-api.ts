import axios from "axios";
import type { ChatRequest, ChatResponse } from "@/types/chat";

const API_BASE = import.meta.env.VITE_API_BASE || "";

const http = axios.create({
  baseURL: API_BASE || undefined,
  timeout: 500000,
});

export async function sendChatMessage(
  request: ChatRequest,
  onEvent?: (event: any) => void,
  signal?: AbortSignal
): Promise<ChatResponse> {
  const formData = new FormData();

  formData.append("command", request.message ?? "");

  // 统一把 images/obj/json 都塞到 files
  const allFiles: File[] = [
    ...(request.images ?? []),
    ...(request.objFiles ?? []),
    ...(request.jsonFiles ?? []),
  ];
  allFiles.forEach((f) => formData.append("files", f));

  // Use fetch for streaming support
  // Note: API_BASE handling needs to be manual if not using axios
  const url = `${API_BASE}/process`;

  try {
    const response = await fetch(url, {
      method: "POST",
      body: formData,
      signal,
    });

    if (!response.ok) {
        const text = await response.text();
        throw new Error(`Server error: ${response.status} ${text}`);
    }

    if (!response.body) {
         throw new Error("No response body");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finalData: any = {};

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
            if (line.startsWith("data: ")) {
                const jsonStr = line.slice(6);
                try {
                    const event = JSON.parse(jsonStr);
                    if (onEvent) onEvent(event);

                    if (event.type === "done") {
                        if (event.content && typeof event.content === 'object') {
                             finalData = event.content;
                        }
                        break;
                    }
                } catch (e) {
                    console.error("Error parsing SSE event:", e);
                }
            }
        }
    }

    // Return a constructed ChatResponse for compatibility
    return {
        reply: finalData.reply || "",
        processed_files: finalData.processed_files || [],
        generated_files: finalData.generated_files || [],
        generated_file: finalData.generated_file,
        file_name: finalData.file_name,
        mime_type: finalData.mime_type
    };

  } catch (error: any) {
    const isCanceled =
      error?.name === "AbortError" ||
      error?.name === "CanceledError" ||
      error?.code === "ERR_CANCELED";
    if (isCanceled) throw error;
    throw new Error(error.message || "Failed to send message");
  }
}

export async function renderJsonToPng(
  input: File,
  signal?: AbortSignal
): Promise<{ image: string }> {
  const fd = new FormData();
  fd.append("json", input);

  const res = await fetch("/api/render", {
    method: "POST",
    body: fd,
    signal,
  });

  const text = await res.text();
  if (!res.ok) {
    throw new Error(`render failed: ${res.status} ${text}`);
  }

  return JSON.parse(text) as { image: string };
}
