/**
 * Types mirroring the Genflow runtime API (`app/agent/runtime_schemas.py`).
 */

export interface RuntimePlan {
  user_intent: string;
  fixed_constraints: Record<string, string>;
  free_variables: string[];
  locked_axes: string[];
  unclear_axes: string[];
  next_action: string;
  clarification_questions: string[];
  reasoning_summary: string;
}

export interface RuntimeSession {
  session_id: string;
  original_intent: string;
  clarified_intent: string;
  clarification_closed: boolean;
  clarification_rounds: number;
  selected_gallery_index: number | null;
  next_action: string;
}

export interface RuntimeCandidate {
  slot: number;
  gallery_index: number;
  id: string;
  image_url: string;
  local_path: string;
  prompt: string;
  negative_prompt: string;
  model: string;
  sampler: string;
  steps: string;
  cfgscale: string;
  seed: string;
  width: number | null;
  height: number | null;
  group_index: number;
  group_label: string;
  distance: number | null;
}

export interface RuntimeWall {
  groups: number[][];
  query_labels: string[];
  candidates: RuntimeCandidate[];
  description: string;
}

export interface RuntimeExpansion {
  label: string;
  prompt: string;
  axis_focus: string[];
  target_cluster_id: number | null;
  checkpoint: string;
  sampler: string;
  loras: string[];
}

export interface RuntimeRecommendation {
  checkpoint: string;
  sampler: string;
  loras: string[];
  reasoning_summary: string;
}

export interface StartResponse {
  session: RuntimeSession;
  plan: RuntimePlan;
}

export interface PlanResponse {
  session: RuntimeSession;
  plan: RuntimePlan;
}

export interface CandidatesResponse {
  session: RuntimeSession;
  plan: RuntimePlan;
  wall: RuntimeWall;
  expansions: RuntimeExpansion[];
  recommendation: RuntimeRecommendation;
}

export interface SelectResponse {
  session: RuntimeSession;
  anchor_summary: string;
  references: Record<string, unknown>[];
  selected_reference_ids: number[];
}

export interface NormalizedSchema {
  prompt: string;
  negative_prompt: string;
  cfgscale: string;
  steps: string;
  sampler: string;
  seed: string;
  model: string;
  clipskip: string;
  style: string[];
  lora: string[];
  full_metadata_string: string;
  raw_fields: Record<string, string>;
}

export interface SchemaResponse {
  session: RuntimeSession;
  normalized: NormalizedSchema;
  raw_metadata: string;
}

export interface ResultResponse {
  session: RuntimeSession;
  payload: Record<string, unknown>;
  summary: Record<string, unknown>;
}

export interface AppliedLora {
  requested: string;
  resolved: string;
  weight: number;
}

export interface WorkflowResponse {
  session: RuntimeSession;
  title: string;
  api_graph: Record<string, unknown>;
  ui_workflow: Record<string, unknown>;
  warnings: string[];
  checkpoint: string;
  requested_checkpoint: string;
  checkpoint_resolved: boolean;
  applied_loras: AppliedLora[];
  unresolved_loras: string[];
  controls: Record<string, unknown>;
  available_checkpoints: string[];
  available_loras: string[];
  remediation: RemediationItem[];
}

export interface PushResponse extends WorkflowResponse {
  pushed: boolean;
  prompt_id: string;
  queue_number: number | null;
  error: string;
  node_errors: Record<string, unknown>;
}

export interface ComfyStatus {
  reachable: boolean;
  base_url: string;
  comfyui_version: string;
  device: string;
  checkpoints: string[];
  loras: string[];
  samplers: string[];
  error: string;
}

export interface GeneratedImage {
  filename: string;
  subfolder: string;
  type: string;
  url: string;
}

export interface PromptResult {
  ready: boolean;
  status?: string;
  images: GeneratedImage[];
  error: string;
}

export interface WorkflowOptions {
  width: number;
  height: number;
  batch_size: number;
  seed: number | null;
  filename_prefix: string;
  /** Explicit checkpoint chosen to work around an uninstalled model. */
  checkpoint_override: string | null;
}

export interface RemediationItem {
  kind: string;
  node_id: string;
  class_type: string;
  input_name: string;
  requested: string;
  installed: string[];
  suggestions: string[];
  fixable: boolean;
  message: string;
}

export interface RefinementHistoryEntry {
  round: number;
  candidates: number[];
  best_slot: number | null;
  worst_slot: number | null;
  skipped: boolean;
}

export interface RefinementState {
  active: boolean;
  finished: boolean;
  seed_indices: number[];
  round_index: number;
  consecutive_skips: number;
  batch_size: number;
  pending_candidates: RuntimeCandidate[];
  history: RefinementHistoryEntry[];
  best_index: number | null;
}

export interface RefineResponse {
  session: RuntimeSession;
  refinement: RefinementState;
  anchor_summary: string;
  selected_reference_ids: number[];
}
