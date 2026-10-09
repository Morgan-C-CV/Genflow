import { useEffect, useState } from "react";
import type { ModifyAxisGroup, ModifyComposition, ModifyProbe, ModifyState } from "../types";

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

/** Distance bands along one axis direction, nearest first. */
const BAND_ORDER = ["near", "mid", "far"] as const;

const BAND_LABEL: Record<string, string> = {
  near: "NEAR",
  mid: "MID",
  far: "FAR",
};

const BAND_CAPTION: Record<string, string> = {
  near: "closest to the current result",
  mid: "halfway along this direction",
  far: "furthest along this direction",
};

/**
 * Groups the flat probe list by axis when the API has not sent axis groups
 * yet, so the picker still renders one block per dissatisfaction axis.
 */
function groupProbesByAxis(probes: ModifyProbe[]): ModifyAxisGroup[] {
  const order: string[] = [];
  const buckets = new Map<string, ModifyProbe[]>();
  for (const probe of probes) {
    const axis = probe.axis || "unlabelled_axis";
    if (!buckets.has(axis)) {
      buckets.set(axis, []);
      order.push(axis);
    }
    buckets.get(axis)!.push(probe);
  }
  return order.map((axis) => {
    const axisProbes = [...buckets.get(axis)!].sort(
      (a, b) => bandRank(a.band) - bandRank(b.band),
    );
    return {
      axis,
      query: "",
      probes: axisProbes,
      selected_probe_id: "",
    };
  });
}

function bandRank(band: string): number {
  const index = BAND_ORDER.indexOf(band as (typeof BAND_ORDER)[number]);
  return index === -1 ? BAND_ORDER.length : index;
}

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
  // PBO pre-selects the top-ranked reference per axis, so a selection can exist
  // before the user has clicked anything.
  const hasProbes = modify.probes.length > 0 || (modify.axis_groups?.length ?? 0) > 0;
  const hasSelection = Boolean(modify.selected_probe_id);
  const hasPreview = Boolean(modify.preview?.probe_id);
  const hasCommit = Boolean(modify.committed_patch?.patch_id);
  const hasExecuted = modify.stage === "executed" || modify.stage === "verified";
  const hasVerified = modify.stage === "verified";

  // `composition` is `{}` until the preview step has composed a unified schema.
  const composition = modify.composition as Partial<ModifyComposition> | undefined;
  const composedSchema = composition?.schema;
  const hasComposition = Boolean(composition?.source || composedSchema?.prompt);

  const axisGroups: ModifyAxisGroup[] =
    modify.axis_groups && modify.axis_groups.length > 0
      ? modify.axis_groups
      : groupProbesByAxis(modify.probes);

  const interpretedBy = modify.interpreted_by;
  const readByLabel =
    interpretedBy === "llm"
      ? "read by model"
      : interpretedBy === "rules"
        ? "read by rules"
        : "reader unknown";

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
            Describe what is wrong and what must stay. The model reads the feedback and
            splits it into modification axes, then retrieves three real gallery
            references per axis at increasing distance — near, mid and far along that
            direction. Pick one reference per axis and the model composes a single
            unified schema. Nothing reaches the committed schema until you approve it.
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

          {/* ---- Σ1/Σ2: parsed feedback, hypotheses, per-axis references ---- */}
          {hasProbes && (
            <section className="modify-block">
              <h3>2 · Reference retrieval per modification axis</h3>

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
                <div className="modify-pills">
                  <span
                    className={`pill interpreted-by ${
                      interpretedBy === "llm" ? "by-llm" : "by-rules"
                    }`}
                  >
                    {readByLabel}
                  </span>
                  <span className="pill">uncertainty {modify.uncertainty.toFixed(2)}</span>
                  <span className="pill">
                    {axisGroups.length} ax{axisGroups.length === 1 ? "is" : "es"} ·{" "}
                    {modify.probes.length} references
                  </span>
                </div>
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

              <div className="axis-groups">
                {axisGroups.map((group) => (
                  <AxisGroupBlock
                    key={group.axis}
                    group={group}
                    selectedProbeId={
                      modify.selected_probe_ids?.[group.axis] || group.selected_probe_id
                    }
                    busy={busy}
                    onSelect={onSelectProbe}
                  />
                ))}
              </div>
              {hasSelection && !hasPreview && (
                <p className="muted small modify-note">
                  Each axis shows three real gallery references along that direction,
                  nearest first. The system pre-picks one per axis — click any other image
                  to override that pick, then preview the composition.
                </p>
              )}
            </section>
          )}

          {/* ---- Σ3 → Σ4: preview ---- */}
          {hasSelection && (
            <section className="modify-block">
              <h3>3 · Preview before it touches the committed graph</h3>
              <p className="muted small">
                Preview composes one unified schema from the reference selected for every
                axis while the committed schema stays exactly as it was — s(Σ4) = s(Σ3).
              </p>
              {!modify.preview.probe_id ? (
                <div className="compose-actions">
                  <button type="button" className="primary" disabled={busy} onClick={onPreview}>
                    {busy ? "Rendering…" : "Preview selected references"}
                  </button>
                </div>
              ) : (
                <>
                  {hasComposition && (
                    <div className="composition">
                      <div className="composition-head">
                        <span
                          className={`pill composition-source ${
                            composition?.source === "llm" ? "by-llm" : "by-rules"
                          }`}
                        >
                          {composition?.source === "llm"
                            ? "schema composed by model"
                            : "schema composed by rules"}
                        </span>
                        <span className="pill unchanged">committed schema unchanged</span>
                      </div>
                      <div className="composition-diff">
                        <span className="axis-label">differs from committed</span>
                        <ul className="chips">
                          {(composition?.differs_from_committed ?? []).length === 0 && (
                            <li className="chip">no differences</li>
                          )}
                          {(composition?.differs_from_committed ?? []).map((field) => (
                            <li key={field} className="chip field-chip">
                              {field}
                            </li>
                          ))}
                        </ul>
                      </div>
                      <div className="composition-field">
                        <span className="axis-label">composed prompt</span>
                        <p className="mono small composition-prompt">
                          {composedSchema?.prompt || "—"}
                        </p>
                      </div>
                      <div className="composition-field">
                        <span className="axis-label">composed negative prompt</span>
                        <p className="mono small composition-prompt">
                          {composedSchema?.negative_prompt || "—"}
                        </p>
                      </div>
                      <dl className="kv composition-kv">
                        <dt>cfg / steps</dt>
                        <dd className="mono">
                          {composedSchema?.cfgscale || "—"} / {composedSchema?.steps || "—"}
                        </dd>
                        <dt>sampler</dt>
                        <dd className="mono">{composedSchema?.sampler || "—"}</dd>
                        <dt>seed</dt>
                        <dd className="mono">{composedSchema?.seed || "—"}</dd>
                        <dt>model</dt>
                        <dd className="mono">{composedSchema?.model || "—"}</dd>
                        <dt>clipskip</dt>
                        <dd className="mono">{composedSchema?.clipskip || "—"}</dd>
                      </dl>
                    </div>
                  )}
                  <div className="modify-outcome">
                    <p className="small">
                      <strong>{modify.preview.probe_id}</strong> —{" "}
                      {modify.preview.summary?.summary_text || "preview rendered"}
                    </p>
                    {(modify.preview.summary?.changed_axes ?? []).length > 0 && (
                      <p className="muted small">
                        changed axes: {modify.preview.summary.changed_axes.join(", ")}
                      </p>
                    )}
                    {(modify.preview.comparison_notes ?? []).map((note: string) => (
                      <p key={note} className="muted small">
                        · {note}
                      </p>
                    ))}
                    <p className="muted small">
                      Nothing is committed yet. Approve the patch below to apply this
                      schema.
                    </p>
                  </div>
                </>
              )}
            </section>
          )}

          {/* ---- Σ4 → Σ5: commit ---- */}
          {hasPreview && !hasComposition && (
            <section className="modify-block">
              <h3>4 · Commit the patch</h3>
              <p className="muted small modify-note">
                The preview step did not report a composed schema, so there is nothing to
                commit yet. Re-run the preview.
              </p>
            </section>
          )}

          {hasPreview && hasComposition && (
            <section className="modify-block">
              <h3>4 · Commit the patch</h3>
              <p className="muted small">
                Committing applies the composed schema above to the committed graph.
              </p>
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

/** One modification axis: three real gallery references along that direction. */
function AxisGroupBlock({
  group,
  selectedProbeId,
  busy,
  onSelect,
}: {
  group: ModifyAxisGroup;
  selectedProbeId: string;
  busy: boolean;
  onSelect: (probeId: string) => void;
}) {
  const selectedProbe = group.probes.find((probe) => probe.probe_id === selectedProbeId);

  return (
    <article className="axis-group" data-axis={group.axis}>
      <header className="axis-group-head">
        <div className="axis-group-title">
          <span className="axis-group-name">{group.axis}</span>
          <span className="axis-group-caption">
            three real gallery references along this direction — these images are the
            preview of this axis
          </span>
        </div>
        <span className="pill axis-group-pick">
          {selectedProbe
            ? `picked ${bandLabel(selectedProbe.band).toLowerCase()} · gallery #${
                selectedProbe.gallery_index
              }`
            : "no reference picked for this axis"}
        </span>
      </header>
      <p className="axis-query muted small">
        {group.query
          ? `retrieval query · “${group.query}”`
          : "no retrieval query reported for this axis"}
      </p>
      <div className="ref-row">
        {group.probes.map((probe) => (
          <ReferenceCard
            key={probe.probe_id}
            probe={probe}
            selected={probe.probe_id === selectedProbeId}
            busy={busy}
            onSelect={onSelect}
          />
        ))}
      </div>
    </article>
  );
}

function bandLabel(band: string): string {
  return BAND_LABEL[band] ?? (band ? band.toUpperCase() : "REF");
}

/** One retrieved gallery reference inside an axis group. */
function ReferenceCard({
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
  const band = probe.band || "";
  const label = bandLabel(band);
  const score = `${probe.score >= 0 ? "+" : ""}${probe.score.toFixed(2)}`;

  return (
    <article
      className={`ref-card ${selected ? "selected" : ""} band-${band || "unknown"}`}
      data-probe-id={probe.probe_id}
    >
      <button
        type="button"
        className="ref-pick"
        disabled={busy || selected}
        aria-pressed={selected}
        title={probe.reference_prompt || probe.summary}
        onClick={() => onSelect(probe.probe_id)}
      >
        <span className={`ref-band band-${band || "unknown"}`}>{label}</span>
        {selected && (
          <span className="candidate-check" aria-hidden="true">
            ✓
          </span>
        )}
        <img
          src={probe.image_url}
          alt={`${label} reference for axis ${probe.axis}, gallery ${probe.gallery_index}`}
          loading="lazy"
        />
        <span className="ref-meta">
          <span className="ref-score mono small">PBO {score}</span>
          <span className="ref-distance mono small">dist {probe.axis_distance.toFixed(1)}</span>
          <span className="ref-alignment mono small">align {probe.alignment.toFixed(2)}</span>
          <span className="ref-index mono small">gallery #{probe.gallery_index}</span>
        </span>
      </button>
      <p className="ref-band-note muted small">
        {BAND_CAPTION[band] ?? "reference along this axis direction"}
      </p>
      <p className="ref-prompt muted small">{probe.reference_prompt || probe.summary}</p>
      <p className="mono small muted ref-source">
        {probe.reference_model || "unknown model"} ·{" "}
        {probe.reference_sampler || "unknown sampler"}
      </p>
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
      <span className={`ref-state ${selected ? "on" : ""}`}>
        {selected ? "selected for this axis" : "click to use this reference"}
      </span>
    </article>
  );
}
