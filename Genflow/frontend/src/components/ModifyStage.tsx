import { useEffect, useState } from "react";
import type { ModifyProbe, ModifyState } from "../types";

/** The Σ transitions of thesis 4.3.4, in order. */
const SIGMA = [
  { key: "baseline", label: "Σ0 feedback" },
  { key: "probes", label: "Σ2 probes" },
  { key: "probe_selected", label: "Σ3 selected" },
  { key: "preview", label: "Σ4 preview" },
  { key: "committed", label: "Σ5 commit" },
  { key: "executed", label: "Σ6 execute" },
  { key: "verified", label: "Σ7 verify" },
] as const;

const REGIME_LABEL: Record<string, string> = {
  close: "close",
  exploratory: "exploratory",
  far: "far",
};

interface ModifyStageProps {
  modify: ModifyState;
  busy: boolean;
  onFeedback: (text: string) => void;
  onSelectProbe: (probeId: string) => void;
  onPreview: () => void;
  onCommit: () => void;
  onExecute: () => void;
  onVerify: () => void;
  onContinue: () => void;
}

export default function ModifyStage({
  modify,
  busy,
  onFeedback,
  onSelectProbe,
  onPreview,
  onCommit,
  onExecute,
  onVerify,
  onContinue,
}: ModifyStageProps) {
  const [draft, setDraft] = useState("");

  useEffect(() => {
    setDraft("");
  }, [modify.round_index]);

  // Gate the wizard on what actually exists rather than on the stage string:
  // PBO pre-selects the top-ranked probe, so a selection can exist before the
  // user has clicked anything.
  const hasProbes = modify.probes.length > 0;
  const hasSelection = Boolean(modify.selected_probe_id);
  const hasPreview = Boolean(modify.preview?.probe_id);
  const hasCommit = Boolean(modify.committed_patch?.patch_id);
  const hasExecuted = modify.stage === "executed" || modify.stage === "verified";
  const hasVerified = modify.stage === "verified";

  const currentIndex = hasVerified
    ? SIGMA.length
    : hasExecuted
      ? 5
      : hasCommit
        ? 4
        : hasPreview
          ? 3
          : hasSelection
            ? 2
            : hasProbes
              ? 1
              : 0;

  const roundLabel = Math.min(modify.round_index + 1, modify.max_rounds);
  const roundsLeft = modify.max_rounds - modify.round_index;

  const baselineIndex = modify.baseline?.gallery_index;
  const baselineSchema = (modify.baseline?.schema ?? {}) as Record<string, string>;

  return (
    <div className="stage modify-stage">
      <div className="stage-head">
        <div>
          <h1>Refine the current result</h1>
          <p className="lede">
            Describe what is wrong and what must stay. The system parses the feedback,
            builds repair hypotheses, then samples three candidates at increasing
            distance — close, exploratory and far — and ranks them by expected value.
            Nothing reaches the committed schema until you approve it.
          </p>
        </div>
        <div className="actions">
          <span className="pill">
            Round <strong>{roundLabel}</strong> / {modify.max_rounds}
          </span>
          <button type="button" className="ghost" onClick={onContinue} disabled={busy}>
            Continue to workflow →
          </button>
        </div>
      </div>

      <ol className="sigma-rail">
        {SIGMA.map((entry, index) => (
          <li
            key={entry.key}
            className={
              index < currentIndex ? "done" : index === currentIndex ? "current" : ""
            }
          >
            {entry.label}
          </li>
        ))}
      </ol>

      <div className="modify-body">
        <aside className="modify-current">
          <h3>Current result</h3>
          {typeof baselineIndex === "number" && (
            <img
              className="anchor-image"
              src={`/api/v1/gallery/image/${baselineIndex}?w=640`}
              alt={`baseline ${baselineIndex}`}
            />
          )}
          <dl className="kv">
            <dt>model</dt>
            <dd>{baselineSchema.model || "—"}</dd>
            <dt>sampler</dt>
            <dd>{baselineSchema.sampler || "—"}</dd>
            <dt>steps / cfg</dt>
            <dd>
              {baselineSchema.steps || "—"} / {baselineSchema.cfgscale || "—"}
            </dd>
          </dl>
          <p className="muted small modify-prompt">{baselineSchema.prompt || ""}</p>
        </aside>

        <div className="modify-steps">
          {/* ---- Σ0: feedback ---- */}
          <section className={`modify-block ${currentIndex > 0 ? "complete" : ""}`}>
            <h3>1 · Feedback on this result</h3>
            <textarea
              className="intent-input modify-input"
              rows={3}
              value={draft}
              placeholder="e.g. it looks too flat and washed out — keep the composition but make the lighting more dramatic"
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
                  event.preventDefault();
                  if (draft.trim()) onFeedback(draft.trim());
                }
              }}
              disabled={busy}
            />
            <div className="compose-actions">
              <button
                type="button"
                className="primary"
                disabled={busy || !draft.trim()}
                onClick={() => onFeedback(draft.trim())}
              >
                {busy ? "Analysing…" : "Analyse feedback"}
              </button>
              {modify.feedback_text && currentIndex > 0 && (
                <span className="muted small">“{modify.feedback_text}”</span>
              )}
            </div>
          </section>

          {/* ---- Σ1/Σ2: parsed feedback, hypotheses, probes ---- */}
          {hasProbes && (
            <section className="modify-block">
              <h3>2 · Hypotheses and ranked probes</h3>

              <div className="modify-parsed">
                <div className="axis">
                  <span className="axis-label">Dissatisfaction axes (D)</span>
                  <ul className="chips">
                    {modify.dissatisfaction_axes.length === 0 && <li className="chip">—</li>}
                    {modify.dissatisfaction_axes.map((axis) => (
                      <li key={axis} className="chip">
                        {axis}
                      </li>
                    ))}
                  </ul>
                </div>
                <div className="axis">
                  <span className="axis-label">Preserve constraints (P)</span>
                  <ul className="chips">
                    {modify.preserve_constraints.length === 0 && <li className="chip">—</li>}
                    {modify.preserve_constraints.map((item) => (
                      <li key={item} className="chip">
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
                <span className="pill">uncertainty {modify.uncertainty.toFixed(2)}</span>
              </div>

              <details className="modify-hypotheses">
                <summary>{modify.hypotheses.length} repair hypotheses</summary>
                <ul>
                  {modify.hypotheses.map((hypothesis) => (
                    <li key={hypothesis.hypothesis_id}>
                      <span className="mono small">{hypothesis.patch_family}</span>
                      <p className="muted small">{hypothesis.summary}</p>
                    </li>
                  ))}
                </ul>
              </details>

              <div className="probe-grid">
                {modify.probes.map((probe) => (
                  <ProbeCard
                    key={probe.probe_id}
                    probe={probe}
                    selected={modify.selected_probe_id === probe.probe_id}
                    busy={busy}
                    onSelect={onSelectProbe}
                  />
                ))}
              </div>
              {hasSelection && !hasPreview && (
                <p className="muted small modify-note">
                  The highest-ranked probe is pre-selected. Pick another card to change
                  it, then preview the change.
                </p>
              )}
            </section>
          )}

          {/* ---- Σ3 → Σ4: preview ---- */}
          {hasSelection && (
            <section className="modify-block">
              <h3>3 · Preview before it touches the committed graph</h3>
              <p className="muted small">
                Preview renders the proposal while the committed schema stays exactly as
                it was — s(Σ4) = s(Σ3).
              </p>
              {!modify.preview.probe_id ? (
                <div className="compose-actions">
                  <button type="button" className="primary" disabled={busy} onClick={onPreview}>
                    {busy ? "Rendering…" : "Preview selected probe"}
                  </button>
                </div>
              ) : (
                <div className="modify-outcome">
                  <p className="small">
                    <strong>{modify.preview.probe_id}</strong> —{" "}
                    {modify.preview.summary?.summary_text || "preview rendered"}
                  </p>
                  {(modify.preview.comparison_notes ?? []).map((note: string) => (
                    <p key={note} className="muted small">
                      · {note}
                    </p>
                  ))}
                </div>
              )}
            </section>
          )}

          {/* ---- Σ4 → Σ5: commit ---- */}
          {hasPreview && (
            <section className="modify-block">
              <h3>4 · Commit the patch</h3>
              {!modify.committed_patch?.patch_id ? (
                <div className="compose-actions">
                  <button type="button" className="primary" disabled={busy} onClick={onCommit}>
                    {busy ? "Committing…" : "Commit patch"}
                  </button>
                </div>
              ) : (
                <div className="modify-outcome">
                  <dl className="kv">
                    <dt>patch</dt>
                    <dd className="mono">{modify.committed_patch.patch_id}</dd>
                    <dt>target fields</dt>
                    <dd className="mono">
                      {(modify.committed_patch.target_fields ?? []).join(", ") || "—"}
                    </dd>
                    <dt>rationale</dt>
                    <dd>{modify.committed_patch.rationale || "—"}</dd>
                  </dl>
                </div>
              )}
            </section>
          )}

          {/* ---- Σ5 → Σ6: execute ---- */}
          {hasCommit && (
            <section className="modify-block">
              <h3>5 · Execute</h3>
              {modify.stage === "committed" ? (
                <div className="compose-actions">
                  <button type="button" className="primary" disabled={busy} onClick={onExecute}>
                    {busy ? "Running…" : "Execute committed patch"}
                  </button>
                </div>
              ) : (
                <p className="muted small">
                  {modify.result?.summary?.summary_text || "executed"}
                </p>
              )}
            </section>
          )}

          {/* ---- Σ6 → Σ7: verify ---- */}
          {hasExecuted && (
            <section className="modify-block">
              <h3>6 · Verify</h3>
              {modify.stage === "executed" ? (
                <div className="compose-actions">
                  <button type="button" className="primary" disabled={busy} onClick={onVerify}>
                    {busy ? "Verifying…" : "Verify result"}
                  </button>
                </div>
              ) : (
                <div className="modify-outcome">
                  <dl className="kv">
                    <dt>improved</dt>
                    <dd>{String(modify.verifier?.improved ?? "—")}</dd>
                    <dt>confidence</dt>
                    <dd>{modify.verifier?.confidence ?? "—"}</dd>
                    <dt>continue</dt>
                    <dd>{String(modify.continue_recommended)}</dd>
                  </dl>
                  <p className="muted small">{modify.verifier?.summary || ""}</p>
                  <div className="compose-actions">
                    {roundsLeft > 0 ? (
                      <span className="muted small">
                        Submit new feedback above to start round {roundLabel + 1}, or
                        continue to the workflow.
                      </span>
                    ) : (
                      <span className="muted small">
                        Reached the {modify.max_rounds}-round cap. Continue to the workflow.
                      </span>
                    )}
                  </div>
                </div>
              )}
            </section>
          )}
        </div>
      </div>
    </div>
  );
}

function ProbeCard({
  probe,
  selected,
  busy,
  onSelect,
}: {
  probe: ModifyProbe;
  selected: boolean;
  busy: boolean;
  onSelect: (probeId: string) => void;
}) {
  return (
    <article className={`probe-card ${selected ? "selected" : ""} regime-${probe.regime}`}>
      <header>
        <span className={`probe-regime ${probe.regime}`}>
          {REGIME_LABEL[probe.regime] ?? (probe.regime || "probe")}
        </span>
        <span className="probe-score mono small">{probe.score >= 0 ? "+" : ""}{probe.score.toFixed(2)}</span>
      </header>
      <p className="probe-summary">{probe.summary}</p>
      <p className="mono small muted">{probe.patch_family}</p>
      <div className="probe-axes">
        <span className="axis-label">targets</span>
        <ul className="chips">
          {probe.target_axes.map((axis) => (
            <li key={axis} className="chip">
              {axis}
            </li>
          ))}
        </ul>
        {probe.preserve_axes.length > 0 && (
          <>
            <span className="axis-label">preserves</span>
            <ul className="chips">
              {probe.preserve_axes.map((axis) => (
                <li key={axis} className="chip">
                  {axis}
                </li>
              ))}
            </ul>
          </>
        )}
      </div>
      {probe.rationale.length > 0 && (
        <details className="probe-rationale">
          <summary>score terms</summary>
          <ul>
            {probe.rationale.map((term) => (
              <li key={term} className="mono small">
                {term}
              </li>
            ))}
          </ul>
        </details>
      )}
      <button
        type="button"
        className={selected ? "ghost small" : "primary"}
        disabled={busy || selected}
        onClick={() => onSelect(probe.probe_id)}
      >
        {selected ? "Selected" : "Select this probe"}
      </button>
    </article>
  );
}
