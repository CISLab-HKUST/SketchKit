<template>
  <div class="input-bar-wrapper">
    <div class="input-bar-container">
      <v-card rounded="xl" elevation="2" class="input-card">
        <!-- Preview block INSIDE the card, on top -->
        <div v-if="selectedFiles.length" class="preview-block">
          <div class="preview-grid">
            <!-- Image/SVG thumbnails -->
            <div
              v-for="p in thumbPreviews"
              :key="p.id"
              class="preview-item preview-item--image"
              :title="p.file.name"
            >
              <img :src="p.url" class="preview-img" :alt="p.file.name" />
              <v-btn
                class="preview-close"
                icon
                size="x-small"
                variant="flat"
                color="surface"
                @click="removeFile(p.index)"
              >
                <v-icon size="14">mdi-close</v-icon>
              </v-btn>
            </div>

            <!-- Non-image files (obj/json/others) -->
            <div
              v-for="f in otherFiles"
              :key="f.id"
              class="preview-item preview-item--file"
              :title="f.file.name"
            >
              <v-icon class="file-icon" size="18">{{
                getFileIcon(f.file)
              }}</v-icon>
              <span class="file-name">{{ f.file.name }}</span>

              <v-btn
                class="preview-close"
                icon
                size="x-small"
                variant="flat"
                color="surface"
                @click="removeFile(f.index)"
              >
                <v-icon size="14">mdi-close</v-icon>
              </v-btn>
            </div>
          </div>
        </div>

        <!-- Input row -->
        <div class="d-flex align-center pa-2">
          <v-btn
            icon
            variant="text"
            size="large"
            :disabled="disabled"
            @click="triggerFileInput"
          >
            <v-icon>mdi-paperclip</v-icon>
          </v-btn>

          <input
            ref="fileInput"
            type="file"
            multiple
            accept="image/*,.svg,.obj,.json"
            class="d-none"
            @change="handleFileSelect"
          />

          <v-textarea
            ref="textareaRef"
            v-model="inputText"
            placeholder="Type your message..."
            variant="plain"
            rows="1"
            auto-grow
            hide-details
            class="flex-grow-1 chat-textarea"
            :disabled="disabled"
            @keydown="onKeydown"
            @compositionstart="onCompositionStart"
            @compositionend="onCompositionEnd"
            @paste="handlePaste"
          />

          <v-btn
            icon
            variant="flat"
            color="primary"
            size="large"
            :disabled="!canSend || disabled"
            @click="handleSend"
          >
            <v-icon>mdi-send</v-icon>
          </v-btn>
        </div>
      </v-card>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, nextTick, watch, onBeforeUnmount } from "vue";

const props = defineProps<{ disabled?: boolean }>();
const emit = defineEmits<{ send: [text: string, files: File[]] }>();

const inputText = ref("");
const selectedFiles = ref<File[]>([]);
const fileInput = ref<HTMLInputElement | null>(null);
const textareaRef = ref<any>(null);
const isComposing = ref(false);

const canSend = computed(() => {
  return (
    (inputText.value.trim().length > 0 || selectedFiles.value.length > 0) &&
    !props.disabled
  );
});

function isSvg(file: File) {
  return (
    file.name.toLowerCase().endsWith(".svg") || file.type === "image/svg+xml"
  );
}

/** ✅ Now: thumbnail supports raster images + SVG */
function isThumbPreviewable(file: File) {
  return file.type.startsWith("image/") || isSvg(file);
}

function getFileIcon(file: File): string {
  if (isSvg(file)) return "mdi-vector-polyline";
  if (file.type.startsWith("image/")) return "mdi-image";
  if (file.name.toLowerCase().endsWith(".obj")) return "mdi-cube-outline";
  if (file.name.toLowerCase().endsWith(".json")) return "mdi-code-json";
  return "mdi-file";
}

function triggerFileInput() {
  fileInput.value?.click();
}

function handleFileSelect(event: Event) {
  const target = event.target as HTMLInputElement;
  if (target.files) selectedFiles.value.push(...Array.from(target.files));
  target.value = "";
}

function handlePaste(event: ClipboardEvent) {
  const items = event.clipboardData?.items;
  if (!items) return;
  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    if (item.type.startsWith("image/")) {
      const file = item.getAsFile();
      if (file) selectedFiles.value.push(file);
    }
  }
}

/** ---------- Previews via ObjectURL ---------- */
type ThumbItem = { id: string; index: number; file: File; url: string };
const thumbPreviews = ref<ThumbItem[]>([]);

function rebuildPreviews() {
  // revoke old URLs
  thumbPreviews.value.forEach((p) => URL.revokeObjectURL(p.url));
  thumbPreviews.value = [];

  selectedFiles.value.forEach((file, index) => {
    if (isThumbPreviewable(file)) {
      thumbPreviews.value.push({
        id: `t_${index}_${file.name}_${file.size}`,
        index,
        file,
        url: URL.createObjectURL(file),
      });
    }
  });
}

const otherFiles = computed(() => {
  return selectedFiles.value
    .map((file, index) => ({
      id: `f_${index}_${file.name}_${file.size}`,
      index,
      file,
    }))
    .filter((x) => !isThumbPreviewable(x.file));
});

watch(
  () => selectedFiles.value,
  () => rebuildPreviews(),
  { deep: true }
);

function removeFile(index: number) {
  selectedFiles.value.splice(index, 1);
}

function clearAll() {
  thumbPreviews.value.forEach((p) => URL.revokeObjectURL(p.url));
  thumbPreviews.value = [];
  selectedFiles.value = [];
  inputText.value = "";
}

function handleSend() {
  if (!canSend.value) return;
  emit("send", inputText.value, selectedFiles.value);
  clearAll();
}

onBeforeUnmount(() => {
  thumbPreviews.value.forEach((p) => URL.revokeObjectURL(p.url));
});

/** ---------- IME + Enter ---------- */
function onCompositionStart() {
  isComposing.value = true;
}
function onCompositionEnd() {
  isComposing.value = false;
}
function onKeydown(e: KeyboardEvent) {
  if (e.key !== "Enter") return;
  if (e.shiftKey) return;
  if ((e as any).isComposing || isComposing.value) return;
  e.preventDefault();
  handleSend();
}

// expose
function setInputText(text: string) {
  inputText.value = text;
  nextTick(() => textareaRef.value?.focus?.());
}
defineExpose({ setInputText });
</script>

<style scoped>
.input-bar-wrapper {
  position: sticky;
  bottom: 0;
  background: transparent !important;
  border-top: none !important;
  padding: 12px 0;
  display: flex;
  justify-content: center;
}

.input-bar-container {
  width: 70vw;
  max-width: 1200px;
  min-width: 500px;
}

/* .input-card {
  background: var(--color-surface);
  overflow: hidden;
} */

.input-card {
  background: #fff !important;
  overflow: hidden;
  border: 1px solid rgba(0, 0, 0, 0.12); /* ✅ 浅灰边框 */
}
.preview-block {
  padding: 10px 10px 8px;
  background: rgba(255, 255, 255, 0.92);
}

.preview-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
}

.preview-item {
  position: relative;
  height: 56px; /* ✅ same height for file + image */
  border-radius: 12px;
  border: 1px solid rgba(0, 0, 0, 0.1);
  background: #fff;
  overflow: hidden;
}

.preview-item--image {
  width: 56px; /* square thumb */
}

.preview-img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.preview-item--file {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 0 34px 0 12px;
  max-width: 360px;
}

.file-icon {
  opacity: 0.85;
}

.file-name {
  font-size: 13px;
  line-height: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.preview-close {
  position: absolute;
  top: 6px;
  right: 6px;
  border-radius: 999px;
  min-width: 22px;
  width: 22px;
  height: 22px;
  padding: 0 !important;
}

@media (max-width: 1024px) {
  .input-bar-container {
    width: 85vw;
  }
}

@media (max-width: 768px) {
  .input-bar-container {
    width: 95vw;
    min-width: 0;
  }

  .preview-item {
    height: 52px;
  }
  .preview-item--image {
    width: 52px;
  }
  .preview-item--file {
    max-width: 240px;
  }
}
</style>
