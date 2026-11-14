<script setup lang="ts">
import type { ChatResponse } from "../types/api";
defineProps<{ result: ChatResponse }>();
function relativeSource(source: string) {
  return /^(?:[a-z]:|[/\\]|[a-z]+:\/\/)/i.test(source) ||
    source.split(/[\\/]/).includes("..")
    ? "Путь скрыт"
    : source;
}
</script>
<template>
  <details class="debug-details">
    <summary>Источники и время · {{ result.sources.length }}</summary>
    <p v-if="result.debug">
      Поиск: {{ result.debug.retrieval_ms }} мс · Генерация:
      {{ result.debug.generation_ms }} мс · Всего:
      {{ result.debug.total_ms }} мс
    </p>
    <p v-else>
      Время не запрашивалось. Включите отладку перед следующим сообщением.
    </p>
    <p v-if="!result.sources.length">Источники не получены.</p>
    <article v-for="source in result.sources" :key="source.chunk_id">
      <strong>{{ source.title }}</strong>
      <p>{{ relativeSource(source.source) }}</p>
      <p class="source-text">{{ source.text }}</p>
    </article>
  </details>
</template>
