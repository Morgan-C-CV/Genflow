import { useEffect, useState } from "react";
import type { RefinementState } from "../types";

interface RefineStageProps {
  refinement: RefinementState;
  busy: boolean;
  maxRounds: number;
  onRunRound: () => void;
  onSubmitFeedback: (bestSlot: number, worstSlot: number) => void;
  onSkipRound: () => void;
  onFinish: () => void;
  onBackToCandidates: () => void;
}

export default function RefineStage({
  refinement,
  busy,
  maxRounds,
  onRunRound,
  onSubmitFeedback,
  onSkipRound,
  onFinish,
  onBackToCandidates,
}: RefineStageProps) {
  const [bestSlot, setBestSlot] = useState<number | null>(null);
  const [worstSlot, setWorstSlot] = useState<number | null>(null);

  const pending = refinement.pending_candidates;
  const pendingKey = pending.map((c) => c.slot).join(",");

  useEffect(() => {
    setBestSlot(null);
    setWorstSlot(null);
  }, [pendingKey]);

  const roundsDone = refinement.history.filter((entry) => !entry.skipped).length;
  const canFinish = roundsDone > 0 || refinement.round_index > 0;

  const markBest = (slot: number) => {
    setBestSlot((current) => (current === slot ? null : slot));
    setWorstSlot((current) => (current === slot ? null : current));
  };

  const markWorst = (slot: number) => {
    setWorstSlot((current) => (current === slot ? null : slot));
    setBestSlot((current) => (current === slot ? null : current));
  };

  return (
    <div className="stage refine-stage">
      <div className="stage-head">
        <div>
          <h1>Refine with preference search</h1>
          <p className="lede">
            Each round proposes {refinement.batch_size} images: two exploit your best
            pick so far, three explore nearby, one explores a distant region. Mark the
            strongest and weakest image to steer the next round.
          </p>
        </div>
        <button
          type="button"
          className="ghost"
          onClick={onBackToCandidates}
          disabled={busy}
        >
          Back to seeds
        </button>
      </div>

      <div className="refine-status">
        <span className="pill">
          Round{" "}
          <strong>
            {Math.min(
              refinement.round_index + (pending.length > 0 ? 1 : 0),
              maxRounds,
            )}
          </strong>{" "}
          / {maxRounds}
        </span>
        <span className="pill">
          {refinement.round_index} completed
        </span>
        <span className="pill">
          Seeds <strong>{refinement.seed_indices.length}</strong>
        </span>
        {refinement.consecutive_skips > 0 && (
          <span className="pill warn">
            {refinement.consecutive_skips} consecutive skip
            {refinement.consecutive_skips === 1 ? "" : "s"} — widening the search
          </span>
        )}
      </div>

      {pending.length === 0 ? (
        <div className="refine-empty">
          <p className="muted">
            {refinement.round_index === 0
              ? "Run the first round to see a fresh batch of candidates."
              : "Submit your feedback or run the next round."}
          </p>
          <div className="actions">
            <button type="button" className="primary" onClick={onRunRound} disabled={busy}>
              {busy ? "Searching…" : `Run round ${refinement.round_index + 1}`}
            </button>
            {canFinish && (
              <button type="button" className="ghost" onClick={onFinish} disabled={busy}>
                Finish and use best match
              </button>
            )}
          </div>
        </div>
      ) : (
        <>
          <div className="candidate-grid refine-grid">
            {pending.map((candidate) => {
              const isBest = bestSlot === candidate.slot;
              const isWorst = worstSlot === candidate.slot;
              return (
                <div
                  key={candidate.gallery_index}
                  className={`candidate-card plain static ${
                    isBest ? "marked-best" : ""
                  } ${isWorst ? "marked-worst" : ""}`}
                >
                  <span className="candidate-slot">#{candidate.slot}</span>
                  <img
                    src={candidate.image_url}
                    alt={`candidate ${candidate.slot}`}
                    loading="lazy"
                  />
                  <div className="mark-row">
                    <button
                      type="button"
                      className={`mark-button best ${isBest ? "active" : ""}`}
                      onClick={() => markBest(candidate.slot)}
                      disabled={busy}
                    >
                      Best
                    </button>
                    <button
                      type="button"
                      className={`mark-button worst ${isWorst ? "active" : ""}`}
                      onClick={() => markWorst(candidate.slot)}
                      disabled={busy}
                    >
                      Worst
                    </button>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="refine-actions">
            <button
              type="button"
              className="primary"
              onClick={() => {
                if (bestSlot !== null && worstSlot !== null) {
                  onSubmitFeedback(bestSlot, worstSlot);
                }
              }}
              disabled={busy || bestSlot === null || worstSlot === null}
              title={
                bestSlot === null || worstSlot === null
                  ? "Mark one image as Best and another as Worst"
                  : undefined
              }
            >
              {busy ? "Submitting…" : "Submit feedback & next round"}
            </button>
            <button type="button" className="ghost" onClick={onSkipRound} disabled={busy}>
              Skip this round
            </button>
            {canFinish && (
              <button type="button" className="ghost" onClick={onFinish} disabled={busy}>
                Finish and use best match
              </button>
            )}
          </div>
        </>
      )}

      {refinement.history.length > 0 && (
        <details className="refine-history" open>
          <summary>{refinement.history.length} rounds recorded</summary>
          <ul>
            {refinement.history.map((entry) => (
              <li key={entry.round}>
                <span className="mono small">Round {entry.round}</span>
                {entry.skipped ? (
                  <span className="warn small"> skipped (penalty recorded)</span>
                ) : (
                  <span className="muted small">
                    {" "}
                    best #{entry.best_slot} · worst #{entry.worst_slot}
                  </span>
                )}
                <span className="mono small muted"> — indices {entry.candidates.join(", ")}</span>
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}
