<script setup lang="ts">
import { ref } from "vue";
import type { LeadRequest } from "../types/api";
defineProps<{ busy: boolean; error: string; disabled: boolean }>();
const emit = defineEmits<{
  submit: [fields: Omit<LeadRequest, "session_id">];
  decline: [];
}>();
const fullName = ref(""),
  phone = ref(""),
  company = ref(""),
  consent = ref(false),
  validation = ref("");
function submit() {
  validation.value = "";
  const digits = phone.value.replace(/\D/g, "");
  if (
    !fullName.value.trim() ||
    !company.value.trim() ||
    !/^[+\d\s()-]+$/.test(phone.value) ||
    digits.length < 7 ||
    digits.length > 15 ||
    !consent.value
  ) {
    validation.value =
      "Заполните ФИО, компанию, телефон (7–15 цифр) и подтвердите согласие.";
    return;
  }
  emit("submit", {
    full_name: fullName.value.trim(),
    phone: phone.value.trim(),
    company: company.value.trim(),
    consent: true,
  });
}
</script>
<template>
  <form class="lead-form" @submit.prevent="submit">
    <div class="eyebrow">СЛЕДУЮЩИЙ ШАГ</div>
    <h3>Сохранить заявку</h3>
    <p>
      Сохраним задачу и контакты внутри прототипа. Используйте только
      вымышленные данные. Внешней отправки нет.
    </p>
    <fieldset :disabled="busy || disabled">
      <label
        >ФИО<input
          v-model="fullName"
          required
          autocomplete="off"
          placeholder="Тестовый Клиент"
      /></label>
      <label
        >Телефон<input
          v-model="phone"
          type="tel"
          required
          autocomplete="off"
          placeholder="+70000000000"
      /></label>
      <label
        >Компания<input
          v-model="company"
          required
          autocomplete="off"
          placeholder="Демо-компания"
      /></label>
      <label class="check"
        ><input v-model="consent" type="checkbox" required /> Согласен сохранить
        указанные данные и диалог в заявке прототипа</label
      >
      <p v-if="validation || error" role="alert" class="error">
        {{ validation || error }}
      </p>
      <div class="actions">
        <button class="primary" type="submit">
          {{ busy ? "Сохраняем…" : "Сохранить заявку" }}</button
        ><button type="button" class="text-button" @click="emit('decline')">
          Продолжить без контактов
        </button>
      </div>
    </fieldset>
  </form>
</template>
