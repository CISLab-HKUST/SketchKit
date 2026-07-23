<template>
  <div class="message-wrapper" :class="message.role">
    <div
      class="message-row"
      :class="message.role === 'user' ? 'is-user' : 'is-assistant'"
    >
      <!-- Avatar -->
      <v-avatar
        :color="message.role === 'user' ? 'primary' : 'grey-lighten-1'"
        size="40"
        class="avatar"
      >
        <v-icon>{{ message.role === "user" ? "mdi-account" : "mdi-robot" }}</v-icon>
      </v-avatar>

      <!-- Bubbles (stack vertically) -->
      <div class="bubble-stack">
        <!-- Attachments (Images + Files) ABOVE text bubble -->
        <v-card
          v-if="hasAnyAttachment"
          class="bubble bubble-attachment"
          variant="flat"
          rounded="xl"
          :elevation="0"
        >
          <div class="attachment-wrap">
            <!-- Images -->
            <div v-if="message.content.images?.length" class="image-grid">
              <div
                v-for="(img, idx) in message.content.images"
                :key="idx"
                class="image-item"
              >
                <v-img :src="img" class="image" cover />
              </div>
            </div>

            <!-- Files -->
            <div
              v-if="message.content.objFiles?.length || message.content.jsonFiles?.length"
              class="file-list"
            >
              <v-chip
                v-for="file in [
                  ...(message.content.objFiles || []),
                  ...(message.content.jsonFiles || []),
                ]"
                :key="file"
                variant="outlined"
                class="ma-1"
              >
                <v-icon start>mdi-file</v-icon>
                {{ file }}
              </v-chip>
            </div>
          </div>
        </v-card>

        <!-- Text bubble (ONLY text + status/result/error/timestamp; NO files/images inside) -->
        <v-card
          v-if="hasAnyTextBubbleContent"
          :class="[
            'bubble',
            'bubble-text',
            message.role === 'user' ? 'user-message' : 'assistant-message',
          ]"
          :elevation="message.role === 'user' ? 0 : 1"
          rounded="xl"
        >
          <div class="message-content">
            <!-- Text Content -->
            <div v-if="message.content.text" class="message-text">
              {{ message.content.text }}
            </div>

            <!-- Processing Status -->
            <div v-if="message.status === 'processing'" class="mt-3">
              <v-progress-linear indeterminate color="primary" />
              <p class="text-caption mt-1 text-grey">
                {{ message.content.statusText || getStatusText(message) }}
              </p>
            </div>

            <!-- Intermediate Results -->
            <IntermediateResults
              v-if="message.content.intermediate_results"
              :results="message.content.intermediate_results"
              class="mt-3"
              @download="$emit('download', $event)"
            />

            <!-- Result Card -->
            <ResultCard
              v-if="message.content.result"
              :result="message.content.result"
              class="mt-3"
              @download="$emit('download', $event)"
            />

            <!-- Error Message -->
            <v-alert
              v-if="message.status === 'error'"
              type="error"
              variant="tonal"
              density="compact"
              class="mt-3"
            >
              {{ message.content.error || "An error occurred" }}
            </v-alert>

            <!-- Timestamp -->
            <div class="message-meta mt-2">
              <span class="text-caption text-grey">
                {{ formatTime(message.timestamp) }}
              </span>
            </div>
          </div>
        </v-card>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";
import type { Message } from "@/types/chat";
import ResultCard from "./ResultCard.vue";
import IntermediateResults from "./IntermediateResults.vue";

const props = defineProps<{
  message: Message;
}>();

defineEmits<{
  download: [payload: { filename?: string; filenames?: string[] }];
  reprocess: [messageId: string];
}>();

const hasAnyAttachment = computed(() => {
  const m = props.message;
  return Boolean(
    (m.content.images && m.content.images.length) ||
      (m.content.objFiles && m.content.objFiles.length) ||
      (m.content.jsonFiles && m.content.jsonFiles.length)
  );
});

const hasAnyTextBubbleContent = computed(() => {
  const m = props.message;
  return Boolean(
    m.content.text ||
      m.content.result ||
      (m.content.intermediate_results && m.content.intermediate_results.length) ||
      m.status === "processing" ||
      m.status === "error"
  );
});

function formatTime(date: Date): string {
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function getStatusText(message: Message): string {
  const intentTexts: Record<string, string> = {
    image2sketch: "Converting to sketch...",
    cleanup: "Cleaning up sketch...",
    vectorization: "Vectorizing...",
    colorization: "Adding colors...",
    animation: "Generating animation...",
    sketch2model: "Creating 3D model...",
  };
  return "Processing...";
}
</script>

<style scoped>
/* --- layout --- */
.message-wrapper {
  width: 100%;
  display: block;
  margin-bottom: 16px;
}

.message-row {
  width: 100%;
  display: flex;
  align-items: flex-start;
  gap: 12px;
}

.message-row.is-user {
  flex-direction: row-reverse; /* avatar on the right */
}

.bubble-stack {
  display: flex;
  flex-direction: column;
  gap: 10px;
  align-items: flex-start;
  width: 100%;
}

.message-row.is-user .bubble-stack {
  align-items: flex-end; /* bubbles stick to right */
}

/* --- bubble sizing behavior --- */
.bubble-text {
  display: inline-block;
  width: fit-content;
  max-width: min(70%, 900px);
}

/* Attachments bubble: no border, no black outline */
.bubble-attachment {
  width: fit-content;
  max-width: min(70%, 900px);
  padding: 10px;
  background: transparent; /* ✅ 不要黑边，也不抢背景 */
  border: none;
  box-shadow: none;
}

/* --- bubble colors --- */
.user-message {
  background: var(--color-user-message);
}

.assistant-message {
  background: var(--color-assistant-message);
}

/* --- avatar --- */
.avatar {
  flex: 0 0 auto;
  margin-top: 2px;
}

/* --- content --- */
.message-content {
  padding: 14px 16px;
  min-width: 0;
}

.message-text {
  white-space: pre-wrap;
  word-break: break-word;
  line-height: 1.6;
}

/* --- attachments --- */
.attachment-wrap {
  display: flex;
  /* flex-direction: column; */
  gap: 10px;
}

/* --- images --- */
.image-grid {
  display: flex;
  /* grid-template-columns: repeat(auto-fill, 150px); */
  gap: 8px;
  justify-content: start;
}

.image-item {
  width: 150px;
  border-radius: 12px;
  overflow: hidden;
}

.image {
  width: 100%;
  height: 100%;
}

/* --- files --- */
.file-list {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
}

/* --- meta --- */
.message-meta {
  display: flex;
  align-items: center;
}

/* --- responsive --- */
@media (max-width: 768px) {
  .bubble-text {
    max-width: 85%;
  }
  .bubble-attachment {
    max-width: 85%;
  }
  .image-grid {
    grid-template-columns: repeat(auto-fill, 120px);
  }
  .image-item {
    width: 120px;
  }
}
</style>
