import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import vuetify from "vite-plugin-vuetify";

export default defineConfig({
  plugins: [vue(), vuetify({ autoImport: true })],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    proxy: {
      "/process": {
        target: "http://10.7.12.88:8000",
        changeOrigin: true,
      },
      "/output": {
        target: "http://10.7.12.88:8000",
        changeOrigin: true,
      },

      "/api": {
        target: "http://10.7.12.88:5000",
        changeOrigin: true,
      },
    },
  },
});
