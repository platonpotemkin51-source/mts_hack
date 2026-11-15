<script setup lang="ts">
import { onMounted, onUnmounted, ref } from "vue";
import { api, friendlyError } from "../api/client";
import type { SessionDetail, SessionSummary } from "../types/api";
import ClientCard from "./ClientCard.vue";
const sessions = ref<SessionSummary[]>([]),
  detail = ref<SessionDetail | null>(null),
  error = ref(""),
  loading = ref(false);
let controller: AbortController | undefined;
let generation = 0;
const formatDate = (value: string) => new Date(value).toLocaleString("ru-RU");
function downloadLead() {
  const lead = detail.value?.lead;
  if (!lead) return;
  const url = URL.createObjectURL(new Blob([lead.summary], { type: "text/plain;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `energytool-${lead.lead_id}.txt`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
async function load(id?: string) {
  const token = ++generation;
  controller?.abort();
  controller = new AbortController();
  loading.value = true;
  error.value = "";
  detail.value = null;
  try {
    if (id) {
      const value = await api.session(id, controller.signal);
      if (token === generation) detail.value = value;
    } else {
      const value = await api.sessions(controller.signal);
      if (token === generation) sessions.value = value;
    }
  } catch (cause) {
    if (token === generation) error.value = friendlyError(cause);
  } finally {
    if (token === generation) loading.value = false;
  }
}
onMounted(() => load());
onUnmounted(() => {
  generation++;
  controller?.abort();
});
</script>
<template>
  <section class="archive-page">
    <div class="section-heading">
      <div>
        <div class="eyebrow">ИСТОРИЯ ОБРАЩЕНИЙ</div>
        <h1>Архив диалогов</h1>
      </div>
      <button @click="load()">Обновить</button>
    </div>
    <p class="muted">
      Все разговоры, в том числе без заявки. Просмотр не возобновляет
      консультацию.
    </p>
    <p v-if="loading" role="status">Загружаем архив…</p>
    <p v-if="error" class="error" role="alert">{{ error }}</p>
    <div class="archive-layout">
      <div class="session-list">
        <p v-if="!sessions.length && !loading && !error">Диалогов пока нет.</p>
        <button
          v-for="(session, index) in sessions"
          :key="session.session_id"
          class="session-item"
          :class="{ selected: detail?.session_id === session.session_id }"
          @click="load(session.session_id)"
        >
          <strong>Диалог {{ sessions.length - index }}</strong
          ><span>{{ formatDate(session.created_at) }}</span
          ><small>{{
            session.status === "closed" ? "Закрыт" : "Активен"
          }}</small>
        </button>
      </div>
      <div v-if="detail" class="archive-detail">
        <h2>Переписка <span class="badge">Только просмотр</span></h2>
        <p>{{ detail.lead_id ? "Заявка сохранена" : "Без заявки" }}</p>
        <section v-if="detail.lead" aria-label="Заявка менеджеру">
          <h3>Заявка менеджеру</h3>
          <p style="white-space: pre-wrap">{{ detail.lead.summary }}</p>
          <button @click="downloadLead">Скачать заявку</button>
        </section>
        <p v-if="!detail.messages.length">Нет сообщений.</p>
        <article
          v-for="(message, index) in detail.messages"
          :key="index"
          class="message"
          :class="message.role"
        >
          <div class="message-author">
            {{ message.role === "user" ? "Вы" : "Консультант" }}
          </div>
          <p>{{ message.content }}</p>
        </article>
        <ClientCard :brief="detail.brief" />
      </div>
      <div v-else-if="!loading" class="archive-placeholder">
        Выберите диалог, чтобы посмотреть переписку и карточку задачи.
      </div>
    </div>
  </section>
</template>
