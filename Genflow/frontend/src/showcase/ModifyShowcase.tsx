/**
 * `/showcase/refine` — walkthrough of the real pipeline, end to end.
 *
 * Runs the genuine Genflow flow before the refinement loop:
 *   planner (LLM) -> clarify (LLM) -> expansions + candidate wall (LLM + embeddings)
 *   -> reference selection -> schema generation (LLM) -> initial result
 *   -> shift/modify loop (thesis 4.3)
 *
 * Nothing here is fabricated: every artifact is produced by the same agent the
 * studio uses, so the walkthrough costs real LLM calls. Execution inside the
 * modify loop still goes through the repo's mock ResultExecutor, which produces
 * payloads and summaries rather than rendered pixels.
 */

import { useCallback, useEffect, useState } from "react";
import { ApiError, api } from "../api";
import CandidatesStage from "../components/CandidatesStage";
import ClarifyStage from "../components/ClarifyStage";
import JsonViewer from "../components/JsonViewer";
import ModifyStage from "../components/ModifyStage";
import { Link } from "../router";
import type {
  ModifyState,
  NormalizedSchema,
  RuntimeCandidate,
  RuntimeExpansion,
  RuntimePlan,
  RuntimeWall,
} from "../types";

const MAX_ROUNDS = 3;

type Phase = "intent" | "clarify" | "candidates" | "schema" | "refine";

const PHASES: { key: Phase; label: string }[] = [
  { key: "intent", label: "1 Intent" },
  { key: "clarify", label: "2 Clarify" },
  { key: "candidates", label: "3 Candidates" },
  { key: "schema", label: "4 Schema" },
  { key: "refine", label: "5 Refine" },
];

const EXAMPLE =
  "cyberpunk rainy night street, neon reflections, a lone figure, cinematic lighting";

export default function ModifyShowcase() {
  const [phase, setPhase] = useState<Phase>("intent");
  const [sessionId, setSessionId] = useState("");
  const [plan, setPlan] = useState<RuntimePlan | null>(null);
  const [wall, setWall] = useState<RuntimeWall | null>(null);
  const [expansions, setExpansions] = useState<RuntimeExpansion[]>([]);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [schema, setSchema] = useState<NormalizedSchema | null>(null);
  const [rawMetadata, setRawMetadata] = useState("");
  const [modify, setModify] = useState<ModifyState | null>(null);

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [activity, setActivity] = useState<string[]>([]);
  const [llmCalls, setLlmCalls] = useState(0);

  const note = useCallback((message: string) => {
    setActivity((previous) => [...previous, message]);
  }, []);

  const describeError = useCallback((caught: unknown, fallback: string): string => {
    if (caught instanceof ApiError) return caught.message;
    if (caught instanceof Error) return caught.message;
    return fallback;
  }, []);

  const loadCandidates = useCallback(
    async (session: string, refresh: boolean) => {
      try {
        const response = await api.candidates(session, {
          refresh,
          per_query_k: 2,
          top_k: 16,
        });
        setPlan(response.plan);
        setWall(response.wall);
        setExpansions(response.expansions);
        setSelectedIndex(null);
        setPhase("candidates");
        setLlmCalls((count) => count + 1);
        note(
          `Expansion model produced ${response.expansions.length} queries and a wall of ${response.wall.candidates.length} images.`,
        );
      } catch (caught) {
        setError(describeError(caught, "Candidate generation failed."));
      }
    },
    [note, describeError],
  );

  const startPipeline = useCallback(
    async (intent: string) => {
      setBusy(true);
      setError("");
      setActivity([]);
      setLlmCalls(0);
      setWall(null);
      setSchema(null);
      setRawMetadata("");
      setModify(null);
      note(`Intent: ${intent}`);
      try {
        const response = await api.startEpisode(intent);
        setSessionId(response.session.session_id);
        setPlan(response.plan);
        setLlmCalls((count) => count + 1);
        note(
          `Planner: next_action=${response.plan.next_action}, locked=${response.plan.locked_axes.join("/") || "none"}.`,
        );

        if (
          response.plan.next_action === "ask_user" &&
          response.plan.clarification_questions.length > 0
        ) {
          setPhase("clarify");
        } else {
          await loadCandidates(response.session.session_id, false);
        }
      } catch (caught) {
        setError(describeError(caught, "The planner failed."));
      } finally {
        setBusy(false);
      }
    },
    [note, describeError, loadCandidates],
  );

  const submitClarification = useCallback(
    async (answers: string[]) => {
      if (!sessionId) return;
      setBusy(true);
      setError("");
      try {
        const response = await api.clarify(sessionId, answers);
        setPlan(response.plan);
        setLlmCalls((count) => count + 1);
        note(`Clarification round ${response.session.clarification_rounds}: closed=${response.session.clarification_closed}.`);

        if (
          response.plan.next_action === "ask_user" &&
          response.plan.clarification_questions.length > 0 &&
          response.session.clarification_rounds < 4
        ) {
          return;
        }
        await loadCandidates(sessionId, false);
      } catch (caught) {
        setError(describeError(caught, "Clarification failed."));
      } finally {
        setBusy(false);
      }
    },
    [sessionId, note, describeError, loadCandidates],
  );

  const confirmReference = useCallback(async () => {
    if (!sessionId || selectedIndex === null) return;
    setBusy(true);
    setError("");
    try {
      note(`Reference: gallery index ${selectedIndex}.`);
      await api.select(sessionId, selectedIndex);

      const schemaResponse = await api.generateSchema(sessionId);
      setSchema(schemaResponse.normalized);
      setRawMetadata(schemaResponse.raw_metadata);
      setLlmCalls((count) => count + 1);
      note(
        `Generation model composed a schema: model=${schemaResponse.normalized.model}, sampler=${schemaResponse.normalized.sampler}.`,
      );

      await api.produceResult(sessionId);
      note("Initial result produced.");

      const modifyState = await api.modifyState(sessionId);
      setModify(modifyState.modify);
      setPhase("refine");
    } catch (caught) {
      setError(describeError(caught, "Schema generation failed."));
    } finally {
      setBusy(false);
    }
  }, [sessionId, selectedIndex, note, describeError]);

  const runModify = useCallback(
    async (
      action: (session: string) => Promise<{ modify: ModifyState }>,
      label: string,
    ) => {
      if (!sessionId) return;
      setBusy(true);
      setError("");
      try {
        const response = await action(sessionId);
        setModify(response.modify);
        note(label);
      } catch (caught) {
        setError(describeError(caught, "The refinement step failed."));
      } finally {
        setBusy(false);
      }
    },
    [sessionId, note, describeError],
  );

  const restart = () => {
    setPhase("intent");
    setSessionId("");
    setPlan(null);
    setWall(null);
    setExpansions([]);
    setSelectedIndex(null);
    setSchema(null);
    setRawMetadata("");
    setModify(null);
    setActivity([]);
    setError("");
    setLlmCalls(0);
  };

  const [draft, setDraft] = useState(EXAMPLE);
  useEffect(() => {
    if (phase === "intent") setDraft(EXAMPLE);
  }, [phase]);

  const phaseIndex = PHASES.findIndex((entry) => entry.key === phase);

  return (
    <div className="showcase">
      <header className="showcase-header">
        <div>
          <h1>Refine — end-to-end walkthrough</h1>
          <p className="muted">
            Runs the real pipeline: the planner, expansion and generation models are
            all called for this session, then the shift/modify loop refines the result
            they produced. Nothing is fabricated, so this walkthrough spends real LLM
            calls. Execution inside the refine loop uses the repo&apos;s mock executor
            (payloads, not pixels).{" "}
            <Link to="/" className="showcase-link">
              ← back to the studio
            </Link>
          </p>
        </div>
      </header>

      <div className="showcase-meter">
        <ol className="sigma-rail">
          {PHASES.map((entry, index) => (
            <li
              key={entry.key}
              className={
                index < phaseIndex ? "done" : index === phaseIndex ? "current" : ""
              }
            >
              {entry.label}
            </li>
          ))}
        </ol>
        <span className="pill">
          LLM calls this run: <strong>{llmCalls}</strong>
        </span>
      </div>

      {error && <div className="showcase-error">{error}</div>}

      {phase === "intent" && (
        <section className="showcase-panel">
          <h2>Step 1 · intent</h2>
          <p className="muted small">
            This is the only place the walkthrough differs from the studio: it starts
            from a textarea instead of a full page, but the session it creates is the
            same one.
          </p>
          <textarea
            className="intent-input"
            rows={3}
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            disabled={busy}
          />
          <div className="compose-actions">
            <button
              type="button"
              className="primary"
              disabled={busy || !draft.trim()}
              onClick={() => void startPipeline(draft.trim())}
            >
              {busy ? "Planning…" : "Run the real pipeline"}
            </button>
            <span className="muted small">1 planner call, then expansions and schema</span>
          </div>
        </section>
      )}

      {plan && phase !== "intent" && (
        <section className="showcase-panel">
          <h2>Planner output</h2>
          <dl className="kv">
            <dt>next action</dt>
            <dd className="mono">{plan.next_action}</dd>
            <dt>locked axes</dt>
            <dd>{plan.locked_axes.join(", ") || "—"}</dd>
            <dt>open axes</dt>
            <dd>{plan.unclear_axes.join(", ") || "—"}</dd>
          </dl>
          {Object.keys(plan.fixed_constraints).length > 0 && (
            <ul className="chips" style={{ marginTop: 10 }}>
              {Object.entries(plan.fixed_constraints).map(([key, value]) => (
                <li key={key} className="chip">
                  {key}: {value}
                </li>
              ))}
            </ul>
          )}
          {plan.reasoning_summary && (
            <p className="muted small" style={{ marginTop: 10 }}>
              {plan.reasoning_summary}
            </p>
          )}
        </section>
      )}

      {phase === "clarify" && plan && (
        <section className="showcase-panel">
          <ClarifyStage
            plan={plan}
            busy={busy}
            onSubmit={(answers) => void submitClarification(answers)}
            onSkip={() => void submitClarification([])}
          />
        </section>
      )}

      {phase === "candidates" && wall && (
        <section className="showcase-panel">
          <CandidatesStage
            wall={wall}
            busy={busy}
            selectedIndex={selectedIndex}
            onSelect={(candidate: RuntimeCandidate) =>
              setSelectedIndex(candidate.gallery_index)
            }
            onRefresh={() => void loadCandidates(sessionId, true)}
            onConfirm={() => void confirmReference()}
          />
          {expansions.length > 0 && (
            <details className="modify-hypotheses" style={{ marginTop: 16 }}>
              <summary>{expansions.length} expansion queries from the LLM</summary>
              <ul>
                {expansions.map((expansion, index) => (
                  <li key={`${index}-${expansion.label}`}>
                    <span className="mono small">
                      {expansion.label} · {expansion.checkpoint || "—"} ·{" "}
                      {expansion.sampler || "—"}
                    </span>
                    <p className="muted small">{expansion.prompt}</p>
                  </li>
                ))}
              </ul>
            </details>
          )}
        </section>
      )}

      {phase === "refine" && modify && (
        <>
          {schema && (
            <section className="showcase-panel">
              <h2>Schema composed by the generation model</h2>
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
              </dl>
              {rawMetadata && (
                <div style={{ marginTop: 12 }}>
                  <JsonViewer
                    value={schema}
                    filename="genflow_schema.json"
                    label="Normalized schema"
                    collapsedHeight={180}
                  />
                </div>
              )}
            </section>
          )}

          <section className="showcase-panel">
            <ModifyStage
              modify={modify}
              busy={busy}
              onFeedback={(text) =>
                void runModify(
                  (id) => api.modifyFeedback(id, text),
                  "Feedback parsed, hypotheses built, three HCS probes sampled.",
                )
              }
              onSelectProbe={(probeId) =>
                void runModify(
                  (id) => api.modifySelect(id, probeId),
                  `Selected probe ${probeId}.`,
                )
              }
              onPreview={() =>
                void runModify(
                  (id) => api.modifyPreview(id),
                  "Preview produced (committed schema untouched).",
                )
              }
              onCommit={() => void runModify((id) => api.modifyCommit(id), "Patch committed.")}
              onExecute={() => void runModify((id) => api.modifyExecute(id), "Patch executed.")}
              onVerify={() => void runModify((id) => api.modifyVerify(id), "Result verified.")}
              onContinue={() => {
                note(
                  `Finished with ${modify.round_index} completed round(s). The studio can push the resulting workflow to ComfyUI.`,
                );
                restart();
              }}
            />
          </section>

          <section className="showcase-panel">
            <p className="muted small">
              The modify loop re-ranks the probes with the PBO scorer each round and
              caps at {MAX_ROUNDS} rounds, matching the user study. Execution inside the
              loop goes through the repo&apos;s mock executor, so it returns payloads and
              summaries rather than rendered pixels; push the workflow from the studio
              to actually render.
            </p>
          </section>
        </>
      )}

      {activity.length > 0 && (
        <section className="showcase-panel">
          <h2>Pipeline activity</h2>
          <ul className="showcase-activity">
            {activity.map((entry, index) => (
              <li key={index}>{entry}</li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
