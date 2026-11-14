<script setup lang="ts">
import { nextTick, onMounted, onUnmounted, ref, watch } from "vue";
import { mockEnabled } from "./api/client";
import { useConversation } from "./state/conversation";
import ClientCard from "./components/ClientCard.vue";
import DebugDetails from "./components/DebugDetails.vue";
import LeadForm from "./components/LeadForm.vue";
import ArchivePanel from "./components/ArchivePanel.vue";

const {
  messages,
  brief,
  busy,
  error,
  draft,
  debug,
  leadOffer,
  declined,
  lead,
  leadBusy,
  leadError,
  uncertain,
  reset,
  send,
  submitLead,
} = useConversation();
const page = ref<"chat" | "archive">("chat"),
  opened = ref(true),
  mobileCard = ref(false),
  showLead = ref(false);
const transcript = ref<HTMLElement>(),
  input = ref<HTMLTextAreaElement>();
const suggestions = [
  "Хочу разобраться с расходами на электроэнергию",
  "Нужно наладить учёт на объекте",
  "Хочу обсудить энергоснабжение склада",
];
function restart() {
  reset();
  showLead.value = false;
  mobileCard.value = false;
  opened.value = true;
  page.value = "chat";
  nextTick(() => input.value?.focus());
}
function closeChat() {
  reset();
  showLead.value = false;
  opened.value = false;
}
function keydown(event: KeyboardEvent) {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    void send();
  }
}
watch(
  () => [messages.value.length, busy.value, showLead.value],
  async () => {
    await nextTick();
    transcript.value?.scrollTo({
      top: transcript.value.scrollHeight,
      behavior: "smooth",
    });
  },
);
watch(lead, (value) => {
  if (value) showLead.value = false;
});
const leave = () => reset();
onMounted(() => window.addEventListener("pagehide", leave));
onUnmounted(() => {
  window.removeEventListener("pagehide", leave);
  reset();
});
</script>

<template>
  <div class="app-shell">
    <aside class="sidebar">
      <a
        href="#"
        class="brand"
        aria-label="Energytool — к чату"
        @click.prevent="page = 'chat'"
        ><span class="brand-mark">мтс</span
        ><span
          >Energytool<span class="brand-caption"
            >ЭНЕРГИЯ ВАШЕГО БИЗНЕСА</span
          ></span
        ></a
      >
      <div class="nav-label">РАБОЧЕЕ ПРОСТРАНСТВО</div>
      <nav aria-label="Основная навигация">
        <button :class="{ active: page === 'chat' }" @click="page = 'chat'">
          <span aria-hidden="true">◫</span> Консультация
          <span class="nav-dot" />
        </button>
        <button
          :class="{ active: page === 'archive' }"
          @click="page = 'archive'"
        >
          <span aria-hidden="true">▤</span> Архив диалогов
        </button>
      </nav>
      <div class="sidebar-bottom">
        <div class="sidebar-note">
          <span class="status-dot" /> Ваш энергетический помощник
          <p>От задачи — к понятному следующему шагу.</p>
        </div>
        <span class="muted">МТС Energytool · Прототип</span>
      </div>
    </aside>
    <main>
      <header class="topbar">
        <span
          >Для бизнеса <span class="separator">/</span> Энергетические
          решения</span
        ><label class="debug-toggle"
          ><input v-model="debug" type="checkbox" /> Отладка</label
        >
      </header>
      <div v-if="mockEnabled" class="mock-banner" role="status">
        ДЕМО / MOCK · Синтетические ответы. Поиск и генерация не подключены.
        Карточка — фиксированный пример склада.
      </div>
      <ArchivePanel v-if="page === 'archive'" />
      <div v-else class="consultation-page">
        <div class="section-heading">
          <div>
            <div class="eyebrow">КОНСУЛЬТАНТ ПО ЭНЕРГЕТИКЕ</div>
            <h1>Разберёмся с вашей задачей<span class="red">.</span></h1>
            <p class="muted">
              Опишите ситуацию. Вместе определим следующий шаг.
            </p>
          </div>
          <button class="new-chat" @click="restart">
            <span aria-hidden="true">＋</span> Новый диалог
          </button>
        </div>
        <div v-if="!opened" class="closed-state">
          <div class="welcome-icon">↗</div>
          <h2>Диалог закрыт</h2>
          <p>
            Предыдущий разговор доступен в архиве. Начните с чистой карточки.
          </p>
          <button class="primary" @click="restart">Открыть чат</button>
        </div>
        <template v-else>
          <div class="mobile-tabs">
            <button
              :class="{ active: !mobileCard }"
              @click="mobileCard = false"
            >
              Диалог</button
            ><button :class="{ active: mobileCard }" @click="mobileCard = true">
              Ваша задача
            </button>
          </div>
          <div class="workspace">
            <section
              class="chat-panel"
              :class="{ 'mobile-hidden': mobileCard }"
              aria-label="Консультация"
            >
              <div class="chat-heading">
                <div class="assistant-avatar">e<span>↗</span></div>
                <div>
                  <strong>Energytool</strong>
                  <div class="assistant-caption">ИИ-консультант</div>
                </div>
                <button
                  class="close-chat"
                  aria-label="Закрыть чат"
                  @click="closeChat"
                >
                  ×
                </button>
              </div>
              <div
                ref="transcript"
                class="transcript"
                role="log"
                aria-label="Сообщения"
                aria-live="polite"
                :aria-busy="busy"
              >
                <div v-if="!messages.length" class="welcome">
                  <div class="welcome-icon">↗</div>
                  <span class="eyebrow">НАЧНЁМ С ГЛАВНОГО</span>
                  <h2>Какая задача<br />стоит перед вами?</h2>
                  <p>
                    Расскажите об объекте и о том, что хотите улучшить. Контакты
                    для консультации не нужны.
                  </p>
                  <div class="suggestions">
                    <button
                      v-for="suggestion in suggestions"
                      :key="suggestion"
                      @click="
                        draft = suggestion;
                        input?.focus();
                      "
                    >
                      {{ suggestion }} <span aria-hidden="true">↗</span>
                    </button>
                  </div>
                </div>
                <article
                  v-for="(message, index) in messages"
                  :key="index"
                  class="message"
                  :class="message.role"
                >
                  <div class="message-author">
                    {{
                      message.role === "user"
                        ? "Вы"
                        : "Energytool · Консультант"
                    }}
                  </div>
                  <p>{{ message.content }}</p>
                  <DebugDetails
                    v-if="debug && message.result"
                    :result="message.result"
                  />
                </article>
                <div v-if="busy" class="waiting" role="status">
                  <span class="pulse" /> Консультант готовит ответ…
                </div>
                <div v-if="error" class="error" role="alert">
                  <p>{{ error }}</p>
                  <p>
                    Статус сохранения сообщения неизвестен. Проверьте архив и
                    начните новый диалог, чтобы избежать повторной отправки.
                  </p>
                  <button @click="page = 'archive'">Проверить архив</button>
                </div>
                <div v-if="lead" class="lead-success" role="status">
                  <span class="eyebrow">ЗАЯВКА СОХРАНЕНА</span>
                  <h3>Контекст задачи собран</h3>
                  <p>{{ lead.summary }}</p>
                  <small
                    >Сохранено внутри прототипа. Внешней отправки нет.</small
                  >
                </div>
                <div
                  v-else-if="leadOffer && !declined && !showLead"
                  class="lead-offer"
                >
                  <h3>Зафиксируем вашу задачу?</h3>
                  <p>
                    Можно сохранить заявку или продолжить консультацию без
                    контактов.
                  </p>
                  <div class="actions">
                    <button
                      :disabled="busy || uncertain"
                      @click="showLead = true"
                    >
                      Оставить заявку</button
                    ><button class="text-button" @click="declined = true">
                      Продолжить без контактов
                    </button>
                  </div>
                </div>
                <LeadForm
                  v-if="showLead && !lead"
                  :busy="leadBusy"
                  :disabled="busy || uncertain"
                  :error="leadError"
                  @submit="submitLead"
                  @decline="
                    declined = true;
                    showLead = false;
                  "
                />
              </div>
              <form class="composer" @submit.prevent="send">
                <label class="sr-only" for="message">Ваше сообщение</label>
                <div class="input-wrap">
                  <textarea
                    id="message"
                    ref="input"
                    v-model="draft"
                    rows="2"
                    :disabled="uncertain"
                    placeholder="Напишите об объекте и вашей задаче…"
                    @keydown="keydown"
                  /><button
                    class="send-button"
                    type="submit"
                    :disabled="!draft.trim() || busy || leadBusy || uncertain"
                    aria-label="Отправить сообщение"
                  >
                    ↑
                  </button>
                </div>
                <div class="composer-footer">
                  <span>Enter — отправить · Shift + Enter — новая строка</span
                  ><button
                    v-if="declined && !lead"
                    type="button"
                    class="text-button"
                    :disabled="busy || uncertain"
                    @click="showLead = true"
                  >
                    Оформить заявку
                  </button>
                </div>
              </form>
            </section>
            <ClientCard
              :brief="brief"
              :class="{ 'mobile-hidden': !mobileCard }"
            />
          </div>
          <p class="footnote">
            Ответы могут требовать уточнения. Условия и рекомендации проверяются
            по базе знаний.
          </p>
        </template>
      </div>
    </main>
  </div>
</template>
