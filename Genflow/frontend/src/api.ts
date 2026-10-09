/**
 * Thin typed client for the Genflow runtime API.
 *
 * Requests are relative by default so the Vite dev-server proxy (`/api` ->
 * http://127.0.0.1:8000) applies. Override with VITE_API_BASE for deployments
 * where the frontend is served from a different origin.
 */

import type {
  CandidatesResponse,
  ComfyStatus,
  ModifyResponse,
  PlanResponse,
  PromptResult,
  PushResponse,
  ResultResponse,
  SchemaResponse,
  SelectResponse,
  StartResponse,
  WorkflowOptions,
  WorkflowResponse,
} from "./types";

export const API_BASE: string =
  (import.meta.env.VITE_API_BASE as string | undefined) ?? "/api/v1";

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...init,
    });
  } catch (cause) {
    throw new ApiError(
      `Cannot reach the backend at ${API_BASE}. Make sure the FastAPI server is running.`,
      0,
    );
  }

  const text = await response.text();
  let body: unknown = null;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = text;
    }
  }

  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    if (body && typeof body === "object" && "detail" in body) {
      const raw = (body as { detail: unknown }).detail;
      detail = typeof raw === "string" ? raw : JSON.stringify(raw);
    } else if (typeof body === "string" && body) {
      detail = body;
    }
    throw new ApiError(detail, response.status);
  }

  return body as T;
}

function workflowBody(options: WorkflowOptions) {
  return JSON.stringify(options);
}

export const api = {
  startEpisode(userIntent: string) {
    return request<StartResponse>("/runtime/episodes", {
      method: "POST",
      body: JSON.stringify({ user_intent: userIntent }),
    });
  },

  clarify(sessionId: string, answers: string[]) {
    return request<PlanResponse>(
      `/runtime/episodes/${encodeURIComponent(sessionId)}/clarify`,
      { method: "POST", body: JSON.stringify({ answers }) },
    );
  },

  candidates(
    sessionId: string,
    options: { refresh?: boolean; per_query_k?: number; top_k?: number } = {},
  ) {
    return request<CandidatesResponse>(
      `/runtime/episodes/${encodeURIComponent(sessionId)}/candidates`,
      {
        method: "POST",
        body: JSON.stringify({
          refresh: options.refresh ?? false,
          per_query_k: options.per_query_k ?? 2,
          top_k: options.top_k ?? 12,
        }),
      },
    );
  },

  select(sessionId: string, galleryIndex: number) {
    return request<SelectResponse>(
      `/runtime/episodes/${encodeURIComponent(sessionId)}/select`,
      { method: "POST", body: JSON.stringify({ gallery_index: galleryIndex }) },
    );
  },

  generateSchema(sessionId: string) {
    return request<SchemaResponse>(
      `/runtime/episodes/${encodeURIComponent(sessionId)}/schema`,
      { method: "POST" },
    );
  },

  produceResult(sessionId: string) {
    return request<ResultResponse>(
      `/runtime/episodes/${encodeURIComponent(sessionId)}/result`,
      { method: "POST" },
    );
  },

  buildWorkflow(sessionId: string, options: WorkflowOptions) {
    return request<WorkflowResponse>(
      `/runtime/episodes/${encodeURIComponent(sessionId)}/workflow`,
      { method: "POST", body: workflowBody(options) },
    );
  },

  pushWorkflow(sessionId: string, options: WorkflowOptions) {
    return request<PushResponse>(
      `/runtime/episodes/${encodeURIComponent(sessionId)}/workflow/push`,
      { method: "POST", body: workflowBody(options) },
    );
  },

  promptResult(promptId: string) {
    return request<PromptResult>(
      `/runtime/workflow/result/${encodeURIComponent(promptId)}`,
    );
  },

  comfyStatus() {
    return request<ComfyStatus>("/runtime/comfyui/status");
  },

  /** Number of records in the gallery. */
  galleryCount() {
    return request<{ total: number }>("/gallery/images?limit=1");
  },

  /** Session snapshot including its plan, used to restore the side panel. */
  episode(sessionId: string) {
    return request<PlanResponse>(
      `/runtime/episodes/${encodeURIComponent(sessionId)}`,
    );
  },

  /**
   * Refine acts on an existing result, so one has to exist before the loop can
   * start. This seeds it from a gallery record: that record's prompt, sampler and
   * model become the current schema and result. Everything after this point is
   * the ordinary modify flow.
   */
  startRefineFromGallery(galleryIndex: number | null) {
    return request<ModifyResponse>("/runtime/showcase/modify", {
      method: "POST",
      body: JSON.stringify({ gallery_index: galleryIndex, label: "Refine" }),
    });
  },

  /* ---------- shift/modify loop (thesis 4.3) ---------- */

  modifyState(sessionId: string) {
    return request<ModifyResponse>(
      `/runtime/episodes/${encodeURIComponent(sessionId)}/modify`,
    );
  },

  modifyFeedback(sessionId: string, feedbackText: string) {
    return request<ModifyResponse>(
      `/runtime/episodes/${encodeURIComponent(sessionId)}/modify/feedback`,
      { method: "POST", body: JSON.stringify({ feedback_text: feedbackText }) },
    );
  },

  modifySelect(sessionId: string, probeId: string) {
    return request<ModifyResponse>(
      `/runtime/episodes/${encodeURIComponent(sessionId)}/modify/select`,
      { method: "POST", body: JSON.stringify({ probe_id: probeId }) },
    );
  },

  modifyPreview(sessionId: string) {
    return request<ModifyResponse>(
      `/runtime/episodes/${encodeURIComponent(sessionId)}/modify/preview`,
      { method: "POST" },
    );
  },

  modifyCommit(sessionId: string) {
    return request<ModifyResponse>(
      `/runtime/episodes/${encodeURIComponent(sessionId)}/modify/commit`,
      { method: "POST" },
    );
  },

  modifyExecute(sessionId: string) {
    return request<ModifyResponse>(
      `/runtime/episodes/${encodeURIComponent(sessionId)}/modify/execute`,
      { method: "POST" },
    );
  },

  modifyVerify(sessionId: string) {
    return request<ModifyResponse>(
      `/runtime/episodes/${encodeURIComponent(sessionId)}/modify/verify`,
      { method: "POST" },
    );
  },
};
