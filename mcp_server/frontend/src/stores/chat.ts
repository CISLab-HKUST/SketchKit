import { defineStore } from "pinia";
import { ref } from "vue";
import type {
  Message,
  ResultData,
  OutputKind,
  ChatResponse,
} from "@/types/chat";
import { sendChatMessage, renderJsonToPng } from "@/api/chat-api";

let abortController: AbortController | null = null;

function filenameFromPath(path: string) {
  const clean = path.split("?")[0].split("#")[0];
  const parts = clean.split("/");
  return parts[parts.length - 1] || `result_${Date.now()}`;
}

function getKindFromPath(path: string, mimeType?: string): OutputKind {
  const mt = (mimeType || "").toLowerCase();
  if (mt.includes("model/gltf-binary") || mt.includes("gltf-binary"))
    return "glb";
  if (mt.includes("image/svg")) return "svg";
  if (mt.includes("application/json")) return "json";
  if (mt.includes("image/gif")) return "gif";
  if (mt.startsWith("image/")) return "image";

  const p = path.toLowerCase();
  if (p.endsWith(".json")) return "json";
  if (p.endsWith(".svg")) return "svg";
  if (p.endsWith(".glb")) return "glb";
  if (p.endsWith(".gif")) return "gif";
  if (p.endsWith(".png") || p.endsWith(".jpg") || p.endsWith(".jpeg"))
    return "image";
  return "unknown";
}

function pickSingleOutputPath(resp: ChatResponse): string | undefined {
  const gf = resp.generated_files;
  const pf = resp.processed_files;
  return (
    resp.generated_file ||
    (gf && gf.length ? gf[gf.length - 1] : undefined) ||
    (pf && pf.length ? pf[pf.length - 1] : undefined)
  );
}

function buildResult(resp: ChatResponse): ResultData | undefined {
  const path = pickSingleOutputPath(resp);
  if (!path) return undefined;

  const file_name = resp.file_name || filenameFromPath(path);
  const kind = getKindFromPath(path, resp.mime_type);

  return {
    type: "file",
    kind,
    path,
    file_name,
    mime_type: resp.mime_type,
    metadata: {
      generated_file: resp.generated_file,
      processed_files: resp.processed_files,
    },
  };
}

export const useChatStore = defineStore("chat", () => {
  const messages = ref<Message[]>([]);
  const isProcessing = ref(false);
  const currentSessionId = ref<string>(generateSessionId());

  async function sendMessage(
    text: string,
    images: File[],
    objFiles: File[],
    jsonFiles: File[]
  ) {
    const controller = new AbortController();
    abortController = controller;
    const intermediateResults: ResultData[] = [];

    // 1) Push user message immediately (for UI)
    const userMessage: Message = {
      id: generateMessageId(),
      role: "user",
      content: {
        text,
        images: images.length ? await filesToBase64(images) : undefined,
        objFiles: objFiles.length ? objFiles.map((f) => f.name) : undefined,
        jsonFiles: jsonFiles.length ? jsonFiles.map((f) => f.name) : undefined,
      },
      timestamp: new Date(),
      status: "done",
    };
    messages.value.push(userMessage);

    // 2) Push assistant placeholder (processing)
    const assistantMessage: Message = {
      id: generateMessageId(),
      role: "assistant",
      content: { text: "" },
      timestamp: new Date(),
      status: "processing",
    };
    messages.value.push(assistantMessage);
    isProcessing.value = true;

    try {
      // Helper to update current assistant message
      const updateMsg = (contentPartial: Partial<any>, status?: string) => {
        const idx = messages.value.findIndex(m => m.id === assistantMessage.id);
        if (idx !== -1) {
          const current = messages.value[idx];
          messages.value[idx] = {
            ...current,
            content: { ...current.content, ...contentPartial },
            status: status ?? current.status
          }
        }
      }

      const pendingTasks: Promise<void>[] = [];

      await sendChatMessage(
        {
          message: text,
          images,
          objFiles,
          jsonFiles,
          context: getRecentMessageIds(5),
        },
        async (event) => {
          // Handle SSE events
          if (event.type === 'status') {
            updateMsg({ statusText: event.content });
          } else if (event.type === 'text') {
            const idx = messages.value.findIndex(m => m.id === assistantMessage.id);
            if (idx !== -1) {
              const currentText = messages.value[idx].content.text || "";
              const newText = currentText ? currentText + "\n\n" + event.content : event.content;
              updateMsg({ text: newText });
            }
          } else if (event.type === 'result') {
            const resPath = event.path;
            const toolName = event.tool;
            const kind = getKindFromPath(resPath);
            const resultItem: ResultData = {
              type: 'file',
              kind,
              path: resPath,
              file_name: filenameFromPath(resPath),
              metadata: { tool: toolName }
            };

            // 1. Push immediately to preserve order
            const itemIndex = intermediateResults.length;
            intermediateResults.push(resultItem);
            
            const idx = messages.value.findIndex(m => m.id === assistantMessage.id);
            if (idx !== -1) {
               updateMsg({ intermediate_results: [...intermediateResults] });
            }

            // 2. If it needs rendering, do it uniquely and async
            if (resultItem.kind === 'json') {
              const task = (async () => {
                try {
                  const jsonFile = await fetchJsonAsFile(resultItem.path, controller.signal);
                  const renderResp = await renderJsonToPng(jsonFile, controller.signal);

                  // Update the item in the local array
                  // We need to create a new object or mutate? 
                  // Let's create new to be safe with reactivity, but need to replace at correct index
                  intermediateResults[itemIndex] = {
                    ...intermediateResults[itemIndex],
                    preview_url: renderResp.image,
                    mime_type: "image/png"
                  };
                  
                  // Update store again
                  updateMsg({ intermediate_results: [...intermediateResults] });
                } catch (e) {
                   if (e instanceof DOMException && e.name === 'AbortError') {
                      // ignore
                   } else {
                      console.warn("Rendering failed", e);
                   }
                }
              })();
              pendingTasks.push(task);
            }
          }
        },
        controller.signal
      );
      
      await Promise.all(pendingTasks);

      // Finalization is handled by the last state of the message.
      // We rely on the stream updates.
      // But we might want to ensure status is 'done'.
      const finalUpdates: any = {};
      if (intermediateResults.length > 0) {
        finalUpdates.result = intermediateResults[intermediateResults.length - 1];
      }
      updateMsg(finalUpdates, "done");

      /* OLD LOGIC commented out/adapted
      const index = messages.value.findIndex(
        (m) => m.id === assistantMessage.id
      );
      const baseResult = buildResult(response);
      ...
      */

      // Rendering JSON preview logic (renderJsonToPng)
      // If the MAIN result is JSON, we can still try to render it.
      // if (idx !== -1) {
      //   const m = messages.value[idx];
      //   if (m.content.result && m.content.result.kind === 'json') {
      //     // Try to render it

      //     const jsonPath = m.content.result.path;
      //     updateMsg({ statusText: "Rendering preview..." }, "processing");
      //     try {
      //       const jsonFile = await fetchJsonAsFile(jsonPath, abortController.signal);
      //       const renderResp = await renderJsonToPng(jsonFile, abortController.signal);

      //       // Update result with preview
      //       const newResult: ResultData = {
      //         ...m.content.result,
      //         preview_url: renderResp.image,
      //         mime_type: "image/png"
      //       };
      //       console.log(newResult);
      //       updateMsg({ result: newResult }, "done");
      //     } catch (e) {
      //       console.warn("Rendering failed", e);
      //       updateMsg({}, "done");
      //     }
      //   }
      // }

      // } catch (error: any) {
    } catch (error: any) {
      console.error("Failed to send message:", error);

      const index = messages.value.findIndex(
        (m) => m.id === assistantMessage.id
      );
      if (index !== -1) {
        const isCanceled =
          error?.name === "AbortError" ||
          error?.name === "CanceledError" ||
          error?.code === "ERR_CANCELED";

        messages.value[index] = {
          ...assistantMessage,
          content: {
            text: isCanceled
              ? "Operation cancelled."
              : error?.message || "Request failed.",
            error: error?.message || String(error),
          },
          status: isCanceled ? "done" : "error",
        };
      }
    } finally {
      isProcessing.value = false;
      abortController = null;
    }
  }

  function stopProcessing() {
    if (abortController) {
      abortController.abort();
      abortController = null;
    }
    isProcessing.value = false;
  }

  function clearMessages() {
    messages.value = [];
    currentSessionId.value = generateSessionId();
  }

  function getRecentMessageIds(count: number): string[] {
    return messages.value.slice(-count).map((m) => m.id);
  }

  return {
    messages,
    isProcessing,
    currentSessionId,
    sendMessage,
    stopProcessing,
    clearMessages,
  };
});

function generateMessageId(): string {
  return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

function generateSessionId(): string {
  return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

async function filesToBase64(files: File[]): Promise<string[]> {
  return Promise.all(
    files.map((file) => {
      return new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result as string);
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });
    })
  );
}
function pickLastJsonPath(resp: ChatResponse): string | undefined {
  const candidates = [
    ...(resp.processed_files || []),
    ...(resp.generated_files || []),
    resp.generated_file,
  ].filter(Boolean) as string[];

  // 从后往前找最后一个 .json
  for (let i = candidates.length - 1; i >= 0; i--) {
    const p = candidates[i];
    if (p.toLowerCase().endsWith(".json")) return p;
  }
  return undefined;
}
async function fetchJsonAsFile(
  pathOrUrl: string,
  signal?: AbortSignal
): Promise<File> {
  // 兼容 "/output/xxx.json" 这种相对路径：用当前 origin 拼成绝对 URL
  const url =
    pathOrUrl.startsWith("http://") || pathOrUrl.startsWith("https://")
      ? pathOrUrl
      : `${window.location.origin}${pathOrUrl.startsWith("/") ? "" : "/"}${pathOrUrl}`;

  const res = await fetch(url, { signal });
  if (!res.ok) {
    throw new Error(`fetch json failed: ${res.status} ${await res.text()}`);
  }

  const blob = await res.blob(); // application/json
  const name = filenameFromPath(pathOrUrl) || "output.json";
  return new File([blob], name, { type: blob.type || "application/json" });
}
