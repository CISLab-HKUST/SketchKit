<template>
  <div v-if="visibleResults.length > 0" class="intermediate-results">
    <v-expansion-panels v-model="panel" density="compact" variant="accordion">
      <v-expansion-panel bg-color="transparent" elevation="0">
        <v-expansion-panel-title class="py-0" style="min-height: 36px">
          <span class="text-caption text-medium-emphasis">
            Intermediate Results ({{ visibleResults.length }})
          </span>
        </v-expansion-panel-title>
        <v-expansion-panel-text>
          <div class="results-container d-flex flex-wrap ga-3">
            <ResultCard
              v-for="(res, idx) in visibleResults"
              :key="idx"
              :result="res"
              :title="res.metadata?.tool || `Step ${idx + 1}`"
              :compact="true"
              @download="$emit('download', $event)"
              class="flex-1"
              style="width: 180px;"
            />
          </div>
        </v-expansion-panel-text>
      </v-expansion-panel>
    </v-expansion-panels>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from "vue";
import type { ResultData } from "@/types/chat";
import ResultCard from "./ResultCard.vue";

const props = defineProps<{
  results: ResultData[];
}>();

defineEmits<{
  download: [payload: { path: string; file_name?: string }];
}>();

const panel = ref(0);

const visibleResults = computed(() => {
  if (!props.results || props.results.length === 0) return [];
  // Exclude the last item as it is shown in the main result card
  return props.results.slice(0, -1);
});
</script>

<style scoped>
.intermediate-results {
  width: 100%;
  margin-top: 4px;
}
.results-container {
    padding-top: 4px;
}
</style>
