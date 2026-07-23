<script setup lang="ts">
import { RouterView } from "vue-router";
</script>

<template>
  <v-app>
    <v-main>
      <router-view />
    </v-main>
  </v-app>
</template>

<style>
/* 设置全局背景为白色 */
html,
body,
#app {
  background: #fff !important;
}
</style>
