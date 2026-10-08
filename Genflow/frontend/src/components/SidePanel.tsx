import { useEffect, useRef } from "react";
import type { ComfyStatus, RuntimePlan } from "../types";

export interface TranscriptEntry {
  id: string;
  role: "user" | "agent" | "system";
  text: string;
  detail?: string;
  tone?: "normal" | "error" | "muted";
}

interface SidePanelProps {
  transcript: TranscriptEntry[];
  plan: RuntimePlan | null;
  status: ComfyStatus | null;
  onRetryStatus: () => void;
}

export default function SidePanel({
  transcript,
  plan,
  status,
  onRetryStatus,
}: SidePanelProps) {
  const endRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [transcript.length]);

  return (
    <aside className="side-panel">
      <section className="panel-block comfy-status">
        <h2>ComfyUI</h2>
        {status === null ? (
          <p className="muted">Detecting…</p>
        ) : status.reachable ? (
          <>
            <p className="status-line ok">
              <span className="dot ok" />
              Connected · {status.device || "unknown"}
            </p>
            <dl className="kv">
              <dt>Version</dt>
              <dd>{status.comfyui_version || "—"}</dd>
              <dt>Address</dt>
              <dd className="mono">{status.base_url}</dd>
            </dl>
          </>
        ) : (
          <>
            <p className="status-line bad">
              <span className="dot bad" />
              Not connected
            </p>
            <p className="warn small">{status.error || "ComfyUI is not responding."}</p>
            <button type="button" className="ghost small" onClick={onRetryStatus}>
              Retry
            </button>
          </>
        )}
      </section>

      <section className="panel-block plan-block">
        <h2>Agent plan</h2>
        {plan === null ? (
          <p className="muted">Not started yet.</p>
        ) : (
          <>
            <p className="plan-action">
              Next action: <strong>{plan.next_action || "—"}</strong>
            </p>
            {plan.reasoning_summary && (
              <p className="muted small">{plan.reasoning_summary}</p>
            )}
            <AxisList label="Locked" values={plan.locked_axes} />
            <AxisList label="Open" values={plan.unclear_axes} />
            {Object.keys(plan.fixed_constraints).length > 0 && (
              <div className="axis">
                <span className="axis-label">Fixed constraints</span>
                <ul className="chips">
                  {Object.entries(plan.fixed_constraints).map(([key, value]) => (
                    <li key={key} className="chip">
                      {key}: {value}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </>
        )}
      </section>

      <section className="panel-block transcript-block">
        <h2>Conversation</h2>
        <div className="transcript">
          {transcript.length === 0 && <p className="muted">No messages yet.</p>}
          {transcript.map((entry) => (
            <article
              key={entry.id}
              className={`msg msg-${entry.role} ${entry.tone ?? "normal"}`}
            >
              <span className="msg-role">
                {entry.role === "user"
                  ? "You"
                  : entry.role === "agent"
                    ? "Agent"
                    : "System"}
              </span>
              <p className="msg-text">{entry.text}</p>
              {entry.detail && <p className="msg-detail">{entry.detail}</p>}
            </article>
          ))}
          <div ref={endRef} />
        </div>
      </section>
    </aside>
  );
}

function AxisList({ label, values }: { label: string; values: string[] }) {
  if (values.length === 0) return null;
  return (
    <div className="axis">
      <span className="axis-label">{label}</span>
      <ul className="chips">
        {values.map((value) => (
          <li key={value} className="chip">
            {value}
          </li>
        ))}
      </ul>
    </div>
  );
}
