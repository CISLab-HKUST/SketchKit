<template>
  <v-card variant="outlined" rounded="lg" class="result-card">
    <v-card-title
      :class="[
        'd-flex align-center',
        compact ? 'pa-2 pb-1 text-body-2' : 'pa-3 pb-2 text-subtitle-1',
      ]"
    >
      <v-icon
        :icon="icon"
        :size="compact ? 'x-small' : 'small'"
        class="mr-2"
      />
      <span class="font-weight-medium text-truncate">{{ title }}</span>
      <v-spacer />
      <v-btn
        variant="text"
        :density="compact ? 'compact' : 'comfortable'"
        :icon="compact"
        :prepend-icon="compact ? undefined : 'mdi-download'"
        @click="download"
      >
        <v-icon v-if="compact" size="small">mdi-download</v-icon>
        <span v-else>Download</span>
      </v-btn>
    </v-card-title>

    <v-divider />

    <v-card-text :class="['pa-3', compact ? 'pt-2' : '']">
      <div v-if="!compact" class="text-caption mb-2">
        <span class="font-weight-medium">{{ displayName }}</span>
        <span v-if="result.kind !== 'json'" class="ml-2"
          >({{ result.kind }})</span
        >
      </div>

      <!-- JSON: no preview -->
      <div v-if="result.kind === 'json'" class="preview-wrap">
        <div v-if="result.preview_url">
          <v-img
            :src="result.preview_url"
            class="preview-image"
            contain
            :max-height="compact ? 120 : 360"
          />

          <div v-if="!compact" class="text-caption text-medium-emphasis mt-2">
            Preview rendered from JSON (download still provides JSON).
          </div>
        </div>
      </div>

      <!-- Images (png/jpg/jpeg) & GIF -->
      <div
        v-else-if="result.kind === 'image' || result.kind === 'gif'"
        class="preview-wrap"
      >
        <v-img
          :src="resolvedUrl"
          class="preview-image"
          contain
          :max-height="compact ? 120 : 360"
        />
      </div>

      <!-- SVG -->
      <div v-else-if="result.kind === 'svg'" class="preview-wrap">
        <img :src="resolvedUrl" class="preview-image" alt="SVG preview" />
      </div>

      <!-- GLB -->
      <div v-else-if="result.kind === 'glb'" class="preview-wrap">
        <model-viewer
          :src="resolvedUrl"
          style="width: 100%;"
          :style="{ height: compact ? '120px' : '360px' }"
          camera-controls
          auto-rotate
          interaction-prompt="none"
          shadow-intensity="1"
          exposure="1"
        />
        <div v-if="!compact" class="text-caption text-medium-emphasis mt-2">
          If the preview is blank, add model-viewer to your index.html (see
          comment above).
        </div>
      </div>

      <!-- Unknown -->
      <div v-else class="text-body-2 text-medium-emphasis">
        No preview available.
      </div>
    </v-card-text>
  </v-card>
</template>

<script setup lang="ts">
import { computed } from "vue";
import type { ResultData } from "@/types/chat";

const props = withDefaults(
  defineProps<{
    result: ResultData;
    title?: string;
    compact?: boolean;
  }>(),
  {
    title: "Result",
    compact: false,
  }
);

const emit = defineEmits<{
  download: [payload: { path: string; file_name?: string }];
}>();

const API_BASE = import.meta.env.VITE_API_BASE || "";

function resolveUrl(path: string) {
  if (/^https?:\/\//i.test(path)) return path;
  return API_BASE ? new URL(path, API_BASE).toString() : path;
}

const resolvedUrl = computed(() => resolveUrl(props.result.path));
const displayName = computed(() => props.result.file_name || props.result.path);

const icon = computed(() => {
  switch (props.result.kind) {
    case "image":
    case "gif":
      return "mdi-image-outline";
    case "svg":
      return "mdi-vector-polyline";
    case "glb":
      return "mdi-cube-outline";
    case "json":
      return "mdi-code-json";
    default:
      return "mdi-file-outline";
  }
});

function download() {
  emit("download", {
    path: props.result.path,
    file_name: props.result.file_name,
  });
}
</script>

<style scoped>
.result-card {
  margin-top: 8px;
}

.preview-wrap {
  width: 100%;
  max-width: 560px;
  margin: 0 auto;
  background: #fafafa;
  border-radius: 10px;
  overflow: hidden;
  border: 1px solid rgba(0, 0, 0, 0.08);
}

.preview-image {
  display: block;
  width: 100%;
  height: auto;
}
</style>
