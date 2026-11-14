<script setup lang="ts">
import type { ClientBrief } from "../types/api";
defineProps<{ brief: ClientBrief }>();
const groups = [
  ["known_parameters", "Известные параметры"],
  ["recommended_products", "Подходящие решения"],
  ["missing_fields", "Что ещё уточнить"],
] as const;
</script>

<template>
  <section class="client-card" aria-label="Карточка клиента">
    <div class="eyebrow">КОНТЕКСТ КОНСУЛЬТАЦИИ</div>
    <h2>Ваша задача <span class="small-symbol">↗</span></h2>
    <p class="muted">Здесь появится то, что консультант узнал из разговора.</p>
    <dl>
      <dt>Объект</dt>
      <dd :class="{ muted: !brief.object }">
        {{ brief.object || "Пока не указан" }}
      </dd>
      <dt>Задача</dt>
      <dd :class="{ muted: !brief.problem }">
        {{ brief.problem || "Пока не описана" }}
      </dd>
    </dl>
    <section v-for="[key, label] in groups" :key="key" class="brief-group">
      <h3>{{ label }}</h3>
      <ul v-if="brief[key].length">
        <li v-for="item in brief[key]" :key="item">{{ item }}</li>
      </ul>
      <p v-else class="muted">
        {{
          key === "recommended_products"
            ? "После уточнения задачи"
            : "Пока нет данных"
        }}
      </p>
    </section>
    <div class="card-note">
      Карточка обновляется по ответам консультанта. Параметры можно исправить в
      чате.
    </div>
  </section>
</template>
