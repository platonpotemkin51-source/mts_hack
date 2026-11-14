import { ref } from "vue";
import { api, friendlyError } from "../api/client";
import type {
  ChatResponse,
  ClientBrief,
  LeadRequest,
  LeadResponse,
} from "../types/api";

export const emptyBrief = (): ClientBrief => ({
  object: null,
  problem: null,
  known_parameters: [],
  recommended_products: [],
  missing_fields: [],
});
export function useConversation() {
  const sessionId = ref<string | null>(null);
  const messages = ref<
    Array<{
      role: "user" | "assistant";
      content: string;
      result?: ChatResponse;
    }>
  >([]);
  const brief = ref(emptyBrief());
  const busy = ref(false),
    error = ref(""),
    draft = ref(""),
    debug = ref(false);
  const leadOffer = ref(false),
    declined = ref(false),
    lead = ref<LeadResponse | null>(null);
  const leadBusy = ref(false),
    leadError = ref("");
  // An ambiguous network failure must not automatically replay a possibly saved message.
  const uncertain = ref(false);
  let generation = 0;
  let leadController: AbortController | undefined;

  const close = (id: string) => {
    void api.close(id).catch(() => {
      /* best effort; archive is retained */
    });
  };
  function reset() {
    generation++;
    leadController?.abort();
    if (sessionId.value) close(sessionId.value);
    sessionId.value = null;
    messages.value = [];
    brief.value = emptyBrief();
    busy.value = false;
    error.value = "";
    draft.value = "";
    leadOffer.value = false;
    declined.value = false;
    lead.value = null;
    leadBusy.value = false;
    leadError.value = "";
    uncertain.value = false;
  }
  async function send() {
    const message = draft.value.trim();
    if (!message || busy.value || leadBusy.value || uncertain.value) return;
    const token = generation;
    busy.value = true;
    error.value = "";
    draft.value = "";
    messages.value.push({ role: "user", content: message });
    try {
      // Keep receiving an old first response so its newly assigned ID can be closed.
      const result = await api.chat({
        session_id: sessionId.value,
        message,
        debug: debug.value,
      });
      if (token !== generation) {
        close(result.session_id);
        return;
      }
      sessionId.value = result.session_id;
      messages.value.push({
        role: "assistant",
        content: result.answer,
        result,
      });
      brief.value = result.brief;
      leadOffer.value = result.lead_offer;
    } catch (cause) {
      if (token === generation) {
        error.value = friendlyError(cause);
        uncertain.value = true;
      }
    } finally {
      if (token === generation) busy.value = false;
    }
  }
  async function submitLead(fields: Omit<LeadRequest, "session_id">) {
    if (
      !sessionId.value ||
      leadBusy.value ||
      busy.value ||
      lead.value ||
      uncertain.value
    )
      return;
    const token = generation;
    leadBusy.value = true;
    leadError.value = "";
    leadController = new AbortController();
    try {
      const result = await api.lead(
        { ...fields, session_id: sessionId.value },
        leadController.signal,
      );
      if (token === generation) {
        lead.value = result;
        leadOffer.value = false;
      }
    } catch (cause) {
      if (token === generation) leadError.value = friendlyError(cause);
    } finally {
      if (token === generation) leadBusy.value = false;
    }
  }
  return {
    sessionId,
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
  };
}
