/**
 * `/showcase/refine` — an interactive walkthrough of the preference-search
 * (refine) interface.
 *
 * It drives the real refinement endpoints against real gallery images, but skips
 * the planner and retrieval pipeline: the seed wall is whatever you pick. That
 * keeps the page useful as a demo without spending LLM calls.
 */

import { useCallback, useEffect, useState } from "react";
import { ApiError, api } from "../api";
import RefineStage from "../components/RefineStage";
import { Link } from "../router";
import type { GalleryImage, RefinementState, RuntimeSession } from "../types";

const MAX_ROUNDS = 8;
const BATCH_SIZE = 6;
const GALLERY_PAGE = 24;

type Phase = "seeds" | "refine" | "done";

export default function RefineShowcase() {
  const [phase, setPhase] = useState<Phase>("seeds");
  const [images, setImages] = useState<GalleryImage[]>([]);
  const [galleryTotal, setGalleryTotal] = useState(0);
  const [selected, setSelected] = useState<number[]>([]);
  const [offset, setOffset] = useState(0);

  const [session, setSession] = useState<RuntimeSession | null>(null);
  const [refinement, setRefinement] = useState<RefinementState | null>(null);
  const [bestIndex, setBestIndex] = useState<number | null>(null);

  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [activity, setActivity] = useState<string[]>([]);

  const note = useCallback((message: string) => {
    setActivity((previous) => [...previous, message]);
  }, []);

  const describeError = useCallback((caught: unknown, fallback: string): string => {
    if (caught instanceof ApiError) return caught.message;
    if (caught instanceof Error) return caught.message;
    return fallback;
  }, []);

  const loadGallery = useCallback(
    async (nextOffset: number) => {
      setLoading(true);
      setError("");
      try {
        const listing = await api.galleryImages(GALLERY_PAGE, nextOffset);
        setImages(listing.images);
        setGalleryTotal(listing.total);
        setOffset(nextOffset);
      } catch (caught) {
        setError(describeError(caught, "Could not load the gallery."));
      } finally {
        setLoading(false);
      }
    },
    [describeError],
  );

  useEffect(() => {
    void loadGallery(0);
  }, [loadGallery]);

  const toggleSeed = (index: number) => {
    setSelected((previous) =>
      previous.includes(index)
        ? previous.filter((value) => value !== index)
        : [...previous, index],
    );
  };

  const advanceRound = useCallback(
    async (sessionId: string, roundIndex: number) => {
      if (roundIndex >= MAX_ROUNDS) {
        note(`Reached ${MAX_ROUNDS} rounds — finish to see the best match.`);
        return;
      }
      const response = await api.refinementRound(sessionId, BATCH_SIZE);
      setRefinement(response.refinement);
      setSession(response.session);
      note(
        `Round ${response.refinement.round_index + 1}: ${response.refinement.pending_candidates.length} candidates proposed.`,
      );
    },
    [note],
  );

  const startSearch = useCallback(async () => {
    if (selected.length === 0) return;
    setBusy(true);
    setError("");
    setActivity([]);
    try {
      note(`Seeds: ${selected.join(", ")}`);
      const showcase = await api.startShowcaseEpisode(selected);
      setSession(showcase.session);
      const started = await api.startRefinement(
        showcase.session.session_id,
        selected,
      );
      setRefinement(started.refinement);
      setPhase("refine");
      note(
        `Preference search seeded with ${started.refinement.seed_indices.length} image(s).`,
      );
      await advanceRound(showcase.session.session_id, 0);
    } catch (caught) {
      setError(describeError(caught, "Could not start the preference search."));
    } finally {
      setBusy(false);
    }
  }, [selected, note, advanceRound, describeError]);

  const runRound = useCallback(async () => {
    if (!session) return;
    setBusy(true);
    setError("");
    try {
      const response = await api.refinementRound(session.session_id, BATCH_SIZE);
      setRefinement(response.refinement);
      setSession(response.session);
      note(
        `Round ${response.refinement.round_index + 1}: ${response.refinement.pending_candidates.length} candidates proposed.`,
      );
    } catch (caught) {
      setError(describeError(caught, "Could not run a round."));
    } finally {
      setBusy(false);
    }
  }, [session, note, describeError]);

  const submitFeedback = useCallback(
    async (best: number, worst: number) => {
      if (!session) return;
      setBusy(true);
      setError("");
      try {
        const response = await api.refinementFeedback(session.session_id, {
          best_slot: best,
          worst_slot: worst,
        });
        setRefinement(response.refinement);
        setSession(response.session);
        note(`Round ${response.refinement.round_index}: best #${best}, worst #${worst}.`);
        await advanceRound(session.session_id, response.refinement.round_index);
      } catch (caught) {
        setError(describeError(caught, "Could not record the feedback."));
      } finally {
        setBusy(false);
      }
    },
    [session, note, advanceRound, describeError],
  );

  const skipRound = useCallback(async () => {
    if (!session) return;
    setBusy(true);
    setError("");
    try {
      const response = await api.refinementFeedback(session.session_id, {
        skip: true,
      });
      setRefinement(response.refinement);
      setSession(response.session);
      note(
        `Round skipped — ${response.refinement.consecutive_skips} in a row, widening the search.`,
      );
      await advanceRound(session.session_id, response.refinement.round_index);
    } catch (caught) {
      setError(describeError(caught, "Could not skip the round."));
    } finally {
      setBusy(false);
    }
  }, [session, note, advanceRound, describeError]);

  const finish = useCallback(async () => {
    if (!session) return;
    setBusy(true);
    setError("");
    try {
      const response = await api.finishRefinement(session.session_id);
      setRefinement(response.refinement);
      setSession(response.session);
      setBestIndex(response.refinement.best_index);
      setPhase("done");
      note(
        `Preference model fitted — best match is gallery index ${response.refinement.best_index}.`,
      );
    } catch (caught) {
      setError(describeError(caught, "Could not finish the search."));
    } finally {
      setBusy(false);
    }
  }, [session, note, describeError]);

  const restart = () => {
    setPhase("seeds");
    setSession(null);
    setRefinement(null);
    setBestIndex(null);
    setSelected([]);
    setActivity([]);
    setError("");
  };

  return (
    <div className="showcase">
      <header className="showcase-header">
        <div>
          <h1>Refine — preference search showcase</h1>
          <p className="muted">
            Interactive walkthrough of the refine step. Real gallery images and the
            real preference model; the planner and retrieval pipeline are skipped, so
            no LLM calls are spent.{" "}
            <Link to="/" className="showcase-link">
              ← back to the studio
            </Link>
          </p>
        </div>
        <span className="pill">/showcase/refine</span>
      </header>

      {error && <div className="showcase-error">{error}</div>}

      {phase === "seeds" && (
        <section className="showcase-panel">
          <div className="showcase-panel-head">
            <div>
              <h2>Step 1 · pick seed images</h2>
              <p className="muted small">
                Choose one or more images that look closest to what you want. The
                preference model treats them as positives and everything else as
                neutral.
              </p>
            </div>
            <div className="actions">
              <button
                type="button"
                className="ghost"
                disabled={loading || busy}
                onClick={() => void loadGallery((offset + GALLERY_PAGE) % Math.max(galleryTotal - GALLERY_PAGE, 1))}
              >
                {loading ? "Loading…" : "Next page"}
              </button>
              <button
                type="button"
                className="primary"
                disabled={busy || selected.length === 0}
                onClick={() => void startSearch()}
              >
                {busy ? "Starting…" : `Start preference search (${selected.length})`}
              </button>
            </div>
          </div>

          <div className="showcase-grid">
            {images.map((image) => {
              const isSelected = selected.includes(image.index);
              return (
                <button
                  key={image.index}
                  type="button"
                  className={`candidate-card plain ${isSelected ? "selected" : ""}`}
                  onClick={() => toggleSeed(image.index)}
                  disabled={busy}
                  aria-pressed={isSelected}
                >
                  <span className="candidate-slot">#{image.index}</span>
                  <img src={image.url} alt={`gallery ${image.index}`} loading="lazy" />
                  {isSelected && <span className="candidate-check">✓</span>}
                </button>
              );
            })}
          </div>
          <p className="muted small">
            Showing indices {offset}–{offset + images.length - 1} of {galleryTotal}.
          </p>
        </section>
      )}

      {phase === "refine" && refinement && (
        <section className="showcase-panel">
          <RefineStage
            refinement={refinement}
            busy={busy}
            maxRounds={MAX_ROUNDS}
            onRunRound={() => void runRound()}
            onSubmitFeedback={(best, worst) => void submitFeedback(best, worst)}
            onSkipRound={() => void skipRound()}
            onFinish={() => void finish()}
            onBackToCandidates={restart}
          />
        </section>
      )}

      {phase === "done" && (
        <section className="showcase-panel">
          <h2>Step 3 · best match</h2>
          <p className="muted small">
            The Gaussian process was fitted over every rated candidate; its argmax is
            the image the rest of the pipeline would inherit from.
          </p>
          <div className="showcase-best">
            {bestIndex !== null && (
              <img
                className="showcase-best-image"
                src={`/api/v1/gallery/image/${bestIndex}?w=1024`}
                alt={`best match ${bestIndex}`}
              />
            )}
            <dl className="kv">
              <dt>gallery index</dt>
              <dd className="mono">{bestIndex ?? "—"}</dd>
              <dt>rounds completed</dt>
              <dd>{refinement?.round_index ?? 0}</dd>
              <dt>seeds</dt>
              <dd className="mono">{refinement?.seed_indices.join(", ") || "—"}</dd>
            </dl>
          </div>
          <div className="actions" style={{ marginTop: 16 }}>
            <button type="button" className="ghost" onClick={restart}>
              Run the showcase again
            </button>
            <Link to="/" className="ghost link">
              Open the studio
            </Link>
          </div>
        </section>
      )}

      {(activity.length > 0 || refinement) && (
        <section className="showcase-panel">
          <h2>Activity</h2>
          <ul className="showcase-activity">
            {activity.map((entry, index) => (
              <li key={index}>{entry}</li>
            ))}
          </ul>
          {refinement && refinement.history.length > 0 && (
            <table className="showcase-table">
              <thead>
                <tr>
                  <th>round</th>
                  <th>result</th>
                  <th>candidates</th>
                </tr>
              </thead>
              <tbody>
                {refinement.history.map((entry) => (
                  <tr key={entry.round}>
                    <td className="mono">{entry.round}</td>
                    <td>
                      {entry.skipped
                        ? "skipped"
                        : `best #${entry.best_slot} · worst #${entry.worst_slot}`}
                    </td>
                    <td className="mono small">{entry.candidates.join(", ")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
      )}
    </div>
  );
}
