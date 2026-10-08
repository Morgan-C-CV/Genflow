import { useMemo, useState } from "react";
import JsonViewer from "./JsonViewer";
import type {
  ComfyStatus,
  GeneratedImage,
  NormalizedSchema,
  PushResponse,
  RemediationItem,
  RuntimeCandidate,
  WorkflowOptions,
  WorkflowResponse,
} from "../types";

interface WorkflowStageProps {
  schema: NormalizedSchema | null;
  rawMetadata: string;
  anchorSummary: string;
  selectedCandidate: RuntimeCandidate | null;
  options: WorkflowOptions;
  onOptionsChange: (options: WorkflowOptions) => void;
  workflow: WorkflowResponse | null;
  push: PushResponse | null;
  images: GeneratedImage[];
  busy: boolean;
  polling: boolean;
  comfyStatus: ComfyStatus | null;
  onBuild: () => void;
  onPush: () => void;
  onApplyCheckpoint: (checkpoint: string) => void;
  onRefreshResult: () => void;
}

type Tab = "api" | "ui" | "metadata";

export default function WorkflowStage({
  schema,
  rawMetadata,
  anchorSummary,
  selectedCandidate,
  options,
  onOptionsChange,
  workflow,
  push,
  images,
  busy,
  polling,
  comfyStatus,
  onBuild,
  onPush,
  onApplyCheckpoint,
  onRefreshResult,
}: WorkflowStageProps) {
  const [tab, setTab] = useState<Tab>("api");

  // No matching checkpoint/LoRA: still pushable, but say so on hover.
  const pushAnyway = Boolean(workflow && !workflow.checkpoint_resolved);

  const setNumber = (key: keyof WorkflowOptions, raw: string) => {
    const parsed = Number.parseInt(raw, 10);
    if (Number.isNaN(parsed)) return;
    if (key === "seed") {
      onOptionsChange({ ...options, seed: parsed });
      return;
    }
    onOptionsChange({ ...options, [key]: parsed });
  };

  return (
    <div className="stage workflow-stage">
      <div className="stage-head">
        <div>
          <h1>Generate a ComfyUI workflow</h1>
          <p className="lede">
            Genflow has produced the metadata / schema. This step converts it into
            ComfyUI&apos;s API workflow JSON; pushing it queues the job for execution.
          </p>
        </div>
        <div className="actions">
          <button type="button" className="ghost" onClick={onBuild} disabled={busy}>
            {busy ? "Working…" : "Build JSON only"}
          </button>
          <button
            type="button"
            className={`primary push-button ${pushAnyway ? "push-anyway" : ""}`}
            onClick={onPush}
            disabled={busy}
            title={pushAnyway ? "Push anyway" : "Push to ComfyUI queue"}
          >
            {busy ? (
              "Pushing…"
            ) : pushAnyway ? (
              <>
                <span className="label-default">Push to ComfyUI queue</span>
                <span className="label-hover">Push anyway</span>
              </>
            ) : (
              "Push to ComfyUI queue"
            )}
          </button>
        </div>
      </div>

      <div className="workflow-overview">
        <div className="overview-card">
          <h3>Selected reference</h3>
          {selectedCandidate ? (
            <>
              <img
                className="anchor-image"
                src={selectedCandidate.image_url}
                alt="selected reference"
              />
              <p className="small mono">
                gallery_index {selectedCandidate.gallery_index} · id{" "}
                {selectedCandidate.id}
              </p>
              {anchorSummary && <p className="muted small">{anchorSummary}</p>}
            </>
          ) : (
            <p className="muted">Nothing selected.</p>
          )}
        </div>

        <div className="overview-card">
          <h3>Schema → ComfyUI</h3>
          {schema ? (
            <dl className="kv">
              <dt>model</dt>
              <dd>{schema.model || "—"}</dd>
              <dt>sampler</dt>
              <dd>{schema.sampler || "—"}</dd>
              <dt>steps / cfg</dt>
              <dd>
                {schema.steps || "—"} / {schema.cfgscale || "—"}
              </dd>
              <dt>clipskip</dt>
              <dd>{schema.clipskip || "—"}</dd>
              <dt>seed</dt>
              <dd>{schema.seed || "—"}</dd>
              <dt>lora</dt>
              <dd>{schema.lora.join(", ") || "—"}</dd>
            </dl>
          ) : (
            <p className="muted">No schema yet.</p>
          )}

          {workflow && (
            <div className="resolve-state">
              <p className={workflow.checkpoint_resolved ? "ok small" : "bad small"}>
                checkpoint: {workflow.checkpoint || "—"}{" "}
                {workflow.checkpoint_resolved ? "matched" : "not matched"}
              </p>
              {workflow.applied_loras.length > 0 && (
                <p className="ok small">
                  Applied LoRAs:{" "}
                  {workflow.applied_loras
                    .map((lora) => `${lora.resolved}@${lora.weight}`)
                    .join(", ")}
                </p>
              )}
              {workflow.unresolved_loras.length > 0 && (
                <p className="warn small">
                  Missing LoRAs: {workflow.unresolved_loras.join(", ")}
                </p>
              )}
            </div>
          )}
        </div>

        <div className="overview-card">
          <h3>Generation parameters</h3>
          <div className="option-grid">
            <label>
              <span>Width</span>
              <input
                type="number"
                min={64}
                step={8}
                value={options.width}
                onChange={(event) => setNumber("width", event.target.value)}
              />
            </label>
            <label>
              <span>Height</span>
              <input
                type="number"
                min={64}
                step={8}
                value={options.height}
                onChange={(event) => setNumber("height", event.target.value)}
              />
            </label>
            <label>
              <span>Batch</span>
              <input
                type="number"
                min={1}
                max={16}
                value={options.batch_size}
                onChange={(event) => setNumber("batch_size", event.target.value)}
              />
            </label>
            <label>
              <span>Seed</span>
              <input
                type="number"
                min={0}
                value={options.seed ?? ""}
                placeholder="Use schema value"
                onChange={(event) => setNumber("seed", event.target.value)}
              />
            </label>
            <label className="wide">
              <span>Filename prefix</span>
              <input
                type="text"
                value={options.filename_prefix}
                onChange={(event) =>
                  onOptionsChange({ ...options, filename_prefix: event.target.value })
                }
              />
            </label>
          </div>
          {comfyStatus && comfyStatus.checkpoints.length > 0 && (
            <p className="muted small">
              Available checkpoints: {comfyStatus.checkpoints.slice(0, 4).join(", ")}
              {comfyStatus.checkpoints.length > 4 ? " …" : ""}
            </p>
          )}
        </div>
      </div>

      {workflow && workflow.remediation.length > 0 && (
        <div className="remediation">
          <h3>Needs attention</h3>
          <ul>
            {workflow.remediation.map((item, index) => (
              <RemediationRow
                key={`${item.kind}-${item.requested}-${index}`}
                item={item}
                busy={busy}
                onApply={onApplyCheckpoint}
              />
            ))}
          </ul>
        </div>
      )}

      {workflow && workflow.warnings.length > 0 && (
        <div className="warnings">
          <h3>Notes</h3>
          <ul>
            {workflow.warnings.map((warning, index) => (
              <li key={index}>{warning}</li>
            ))}
          </ul>
        </div>
      )}

      {push && (
        <div className={`push-result ${push.pushed ? "ok" : "bad"}`}>
          {push.pushed ? (
            <>
              <h3>Queued in ComfyUI</h3>
              <dl className="kv">
                <dt>prompt_id</dt>
                <dd className="mono">{push.prompt_id}</dd>
                <dt>Queue number</dt>
                <dd>{push.queue_number ?? "—"}</dd>
              </dl>
              <p className="muted small">
                Results appear here once ComfyUI finishes. Current status:{" "}
                {polling ? "polling…" : "idle"}
              </p>
              <div className="actions">
                <button type="button" className="ghost small" onClick={onRefreshResult}>
                  Refresh results
                </button>
                <a
                  className="ghost small link"
                  href={comfyStatus?.base_url ?? "http://127.0.0.1:8188"}
                  target="_blank"
                  rel="noreferrer"
                >
                  Open ComfyUI
                </a>
              </div>
            </>
          ) : (
            <>
              <h3>Push failed</h3>
              <p>{push.error || "Unknown error"}</p>
              {Object.keys(push.node_errors).length > 0 && (
                <JsonViewer
                  value={push.node_errors}
                  filename="node_errors.json"
                  label="ComfyUI node errors"
                  collapsedHeight={160}
                />
              )}
            </>
          )}
        </div>
      )}

      {images.length > 0 && (
        <div className="results">
          <h3>Generated results</h3>
          <div className="result-grid">
            {images.map((image) => (
              <a
                key={image.url}
                href={image.url}
                target="_blank"
                rel="noreferrer"
                className="result-item"
              >
                <img src={image.url} alt={image.filename} />
                <span className="mono small">{image.filename}</span>
              </a>
            ))}
          </div>
        </div>
      )}

      {workflow && (
        <div className="json-section">
          <div className="tabs">
            <button
              type="button"
              className={tab === "api" ? "tab active" : "tab"}
              onClick={() => setTab("api")}
            >
              API workflow JSON
            </button>
            <button
              type="button"
              className={tab === "ui" ? "tab active" : "tab"}
              onClick={() => setTab("ui")}
            >
              Canvas workflow JSON
            </button>
            <button
              type="button"
              className={tab === "metadata" ? "tab active" : "tab"}
              onClick={() => setTab("metadata")}
            >
              Agent raw metadata
            </button>
          </div>

          {tab === "api" &&
            (Object.keys(workflow.api_graph).length > 0 ? (
              <JsonViewer
                value={workflow.api_graph}
                filename="genflow_api_workflow.json"
                label="POST /prompt payload"
              />
            ) : (
              <p className="muted">Not built yet.</p>
            ))}

          {tab === "ui" &&
            (Object.keys(workflow.ui_workflow).length > 0 ? (
              <JsonViewer
                value={workflow.ui_workflow}
                filename="genflow_ui_workflow.json"
                label="Drag into the ComfyUI canvas"
              />
            ) : (
              <p className="muted">
                Canvas format unavailable (needs ComfyUI /object_info).
              </p>
            ))}

          {tab === "metadata" &&
            (rawMetadata ? (
              <JsonViewer
                value={safeParse(rawMetadata)}
                filename="genflow_metadata.json"
                label="Metadata produced by the LLM"
              />
            ) : (
              <p className="muted">Not generated yet.</p>
            ))}
        </div>
      )}
    </div>
  );
}

function RemediationRow({
  item,
  busy,
  onApply,
}: {
  item: RemediationItem;
  busy: boolean;
  onApply: (checkpoint: string) => void;
}) {
  // Offer the closest matches first, then everything else that is installed.
  const choices = useMemo(() => {
    const seen = new Set<string>();
    const list: string[] = [];
    for (const name of [...item.suggestions, ...item.installed]) {
      if (name && !seen.has(name)) {
        seen.add(name);
        list.push(name);
      }
    }
    return list;
  }, [item.suggestions, item.installed]);

  const [choice, setChoice] = useState(choices[0] ?? "");

  return (
    <li className={`remediation-item ${item.kind}`}>
      <p className="remediation-message">
        <span className="remediation-kind">{item.kind}</span>
        {item.message}
      </p>

      {item.fixable && choices.length > 0 ? (
        <div className="remediation-fix">
          <select
            value={choice}
            onChange={(event) => setChoice(event.target.value)}
            disabled={busy}
            aria-label={`Choose an installed ${item.kind}`}
          >
            {choices.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
          <button
            type="button"
            className="ghost small"
            disabled={busy || !choice}
            onClick={() => onApply(choice)}
          >
            Rebuild &amp; push with this checkpoint
          </button>
        </div>
      ) : (
        item.suggestions.length > 0 && (
          <p className="muted small">
            Closest installed: {item.suggestions.join(", ")}
          </p>
        )
      )}
    </li>
  );
}

function safeParse(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}
