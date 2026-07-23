<template>
  <v-app class="chat-container">
    <!-- Header -->
    <v-app-bar elevation="1" color="surface">
      <v-app-bar-title class="d-flex align-center">
        <v-icon icon="mdi-brush-variant" class="mr-2" color="primary" />
        <span class="text-h6 font-weight-medium">SketchKit Agent</span>
      </v-app-bar-title>

      <template #append>
        <v-btn icon variant="text" @click="clearChat">
          <v-icon>mdi-delete-outline</v-icon>
        </v-btn>
        <v-btn icon variant="text" @click="showInfo = !showInfo">
          <v-icon>mdi-information-outline</v-icon>
        </v-btn>
      </template>
    </v-app-bar>

    <!-- Info Dialog -->
    <v-dialog v-model="showInfo" max-width="600">
      <v-card rounded="lg">
        <v-card-title class="d-flex align-center">
          <v-icon icon="mdi-information" class="mr-2" />
          About SketchKit Agent
        </v-card-title>
        <v-card-text>
          <p class="mb-3">
            SketchKit Agent is an AI-powered conversational interface for
            sketch processing.
          </p>
          <v-list density="compact">
            <v-list-item prepend-icon="mdi-image-edit"
              >Image to Sketch conversion</v-list-item
            >
            <v-list-item prepend-icon="mdi-broom">Sketch cleanup</v-list-item>
            <v-list-item prepend-icon="mdi-vector-polyline"
              >Vectorization</v-list-item
            >
            <v-list-item prepend-icon="mdi-palette">Colorization</v-list-item>
            <v-list-item prepend-icon="mdi-animation"
              >Animation generation</v-list-item
            >
            <v-list-item prepend-icon="mdi-cube-outline"
              >3D model generation</v-list-item
            >
          </v-list>
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn color="primary" variant="text" @click="showInfo = false"
            >Close</v-btn
          >
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Messages Area -->
    <v-main>
      <div ref="messagesContainer" class="messages-container">
        <div class="messages-wrapper">
          <div v-if="messages.length === 0" class="empty-state">
            <v-icon size="64" color="grey-lighten-1"
              >mdi-message-text-outline</v-icon
            >
            <h2 class="text-h5 mt-4 mb-2">Welcome to SketchKit Agent</h2>
            <p class="text-body-2 text-grey">
              Upload files and describe what you want to do with it
            </p>

            <div class="suggestion-chips mt-6">
              <v-chip
                v-for="suggestion in suggestions"
                :key="suggestion"
                class="ma-1"
                variant="outlined"
                @click="useSuggestion(suggestion)"
              >
                {{ suggestion }}
              </v-chip>
            </div>
          </div>

          <ChatMessage
            v-for="message in messages"
            :key="message.id"
            :message="message"
            @download="handleDownload"
            @reprocess="handleReprocess"
          />

          <!-- <div
            v-if="isProcessing"
            class="text-center my-4 processing-indicator"
          >
            <v-progress-circular indeterminate color="primary" />
            <p class="text-caption mt-2">Processing...</p>
            <v-btn
              size="small"
              variant="outlined"
              color="error"
              @click="stopProcessing"
              class="mt-2"
            >
              <v-icon start>mdi-stop-circle-outline</v-icon>
              Stop
            </v-btn>
          </div> -->
        </div>
      </div>
    </v-main>

    <!-- Input Area -->
    <ChatInputBar
      ref="chatInputRef"
      :disabled="isProcessing"
      @send="handleSend"
    />
  </v-app>
</template>

<script setup lang="ts">
import { ref, nextTick, watch } from "vue";
import { useChatStore } from "@/stores/chat";
import { storeToRefs } from "pinia";
import ChatMessage from "@/components/chat/ChatMessage.vue";
import ChatInputBar from "@/components/chat/ChatInputBar.vue";

const chatStore = useChatStore();
const { messages, isProcessing } = storeToRefs(chatStore);

const messagesContainer = ref<HTMLElement | null>(null);
const showInfo = ref(false);

const suggestions = [
  "Convert this image to sketch",
  "Clean up this sketch",
  "Vectorize this sketch",
  "Colorize with warm tones",
  "Generate animation",
];

function categorizeFiles(files: File[]) {
  const images: File[] = [];
  const objFiles: File[] = [];
  const jsonFiles: File[] = [];

  files.forEach((file) => {
    if (file.type.startsWith("image/") || file.name.endsWith(".svg")) {
      images.push(file);
    } else if (file.name.endsWith(".obj")) {
      objFiles.push(file);
    } else if (file.name.endsWith(".json")) {
      jsonFiles.push(file);
    }
  });

  return { images, objFiles, jsonFiles };
}

function scrollToBottom() {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight;
    }
  });
}

async function handleSend(text: string, files: File[]) {
  const { images, objFiles, jsonFiles } = categorizeFiles(files);
  await chatStore.sendMessage(text, images, objFiles, jsonFiles);
  scrollToBottom();
}
const API_BASE = import.meta.env.VITE_API_BASE || "";

// path 只会是类似：/output/vectorized/xxx.json
function resolveUrl(path: string) {
  // 允许你将来后端直接给绝对 URL（不影响）
  if (/^https?:\/\//i.test(path)) return path;
  return API_BASE ? new URL(path, API_BASE).toString() : path;
}

function filenameFromPath(path: string) {
  const clean = path.split("?")[0].split("#")[0];
  const parts = clean.split("/");
  return parts[parts.length - 1] || `result_${Date.now()}`;
}

function triggerDownloadByPath(path: string) {
  const url = resolveUrl(path);
  console.log("download url =", resolveUrl(path));

  const filename = filenameFromPath(path);

  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

function handleDownload(payload: { path?: string; paths?: string[] }) {
  if (payload.path) {
    triggerDownloadByPath(payload.path);
    return;
  }
  if (payload.paths?.length) {
    payload.paths.forEach(triggerDownloadByPath);
  }
}

function handleReprocess(messageId: string) {
  // TODO: Implement reprocess logic
  console.log("Reprocess message:", messageId);
}

function stopProcessing() {
  chatStore.stopProcessing();
}

function clearChat() {
  chatStore.clearMessages();
}

const chatInputRef = ref<InstanceType<typeof ChatInputBar> | null>(null);

function useSuggestion(text: string) {
  chatInputRef.value?.setInputText(text);
}

// Auto-scroll when new messages arrive
watch(
  () => messages.value.length,
  () => {
    scrollToBottom();
  }
);
</script>

<style scoped>
.chat-container {
  height: 100vh;
  display: flex;
  flex-direction: column;
}

.messages-container {
  flex: 1;
  display: flex;
  justify-content: center;
}

.messages-wrapper {
  width: 75vw;
  max-width: 1200px;
  min-width: 500px;
  padding-top: 8px;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 60vh;
  text-align: center;
  color: var(--color-text-secondary);
}

.suggestion-chips {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  max-width: 600px;
}

.processing-indicator {
  padding: 16px;
  background: rgba(25, 118, 210, 0.05);
  border-radius: 12px;
  border: 1px solid rgba(25, 118, 210, 0.1);
}

@media (max-width: 1024px) {
  .messages-wrapper {
    width: 85vw;
  }
}

@media (max-width: 768px) {
  .messages-wrapper {
    width: 95vw;
  }
}
/* 让整个应用背景变白（否则默认 theme 是灰） */
:deep(.v-application__wrap) {
  background: #fff !important;
}

:deep(.v-main) {
  background: #fff !important;
}
</style>
