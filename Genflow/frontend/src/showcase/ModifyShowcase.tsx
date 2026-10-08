/**
 * `/showcase/refine` — walkthrough of the shift/modify refinement loop.
 *
 * Runs the real Σ state machine (thesis 4.3) against a baseline taken from a
 * gallery record: feedback → hypotheses → three HCS probes → preview → commit →
 * execute → verify, capped at three rounds. The create path is skipped, so no
 * LLM calls are spent.
 */

import { useCallback, useEffect, useState } from "react";
import { ApiError, api } from "../api";
import ModifyStage from "../components/ModifyStage";
import { Link } from "../router";
import type { GalleryImage, ModifyState } from "../types";

const GALLERY_PAGE = 24;

type Phase = "baseline" | "refine";

export default function ModifyShowcase() {
  const [phase, setPhase] = useState<Phase>("baseline");
  const [images, setImages] = useState<GalleryImage[]>([]);
  const [galleryTotal, setGalleryTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [baselineIndex, setBaselineIndex] = useState<number | null>(null);

  const [sessionId, setSessionId] = useState("");
  const [modify, setModify] = useState<ModifyState | null>(null);

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

  const startWithBaseline = useCallback(async () => {
    if (baselineIndex === null) return;
    setBusy(true);
    setError("");
    setActivity([]);
    try {
      note(`Baseline: gallery index ${baselineIndex}`);
      const response = await api.startShowcaseModify(baselineIndex, "Modify showcase");
      setSessionId(response.session.session_id);
      setModify(response.modify);
      setPhase("refine");
      note("Baseline schema loaded from the gallery record.");
    } catch (caught) {
      setError(describeError(caught, "Could not create the baseline session."));
    } finally {
      setBusy(false);
    }
  }, [baselineIndex, note, describeError]);

  /** Wrap a modify API call with the showcase's session and activity log. */
  const run = useCallback(
    async (
      action: (sessionId: string) => Promise<{ modify: ModifyState }>,
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
    setPhase("baseline");
    setSessionId("");
    setModify(null);
    setBaselineIndex(null);
    setActivity([]);
    setError("");
  };

  return (
    <div className="showcase">
      <header className="showcase-header">
        <div>
          <h1>Refine — shift/modify showcase</h1>
          <p className="muted">
            Walks the shift/modify state machine from the thesis (feedback →
            hypotheses → three HCS probes → preview → commit → execute → verify, at
            most three rounds). Real gallery images and the real ranking pipeline; the
            create path is skipped, so no LLM calls are spent.{" "}
            <Link to="/" className="showcase-link">
              ← back to the studio
            </Link>
          </p>
        </div>
        <span className="pill">/showcase/refine</span>
      </header>

      {error && <div className="showcase-error">{error}</div>}

      {phase === "baseline" && (
        <section className="showcase-panel">
          <div className="showcase-panel-head">
            <div>
              <h2>Step 1 · pick the result to modify</h2>
              <p className="muted small">
                The refine loop starts from an existing result. Pick a gallery image to
                use as the baseline; its prompt, sampler and model become the committed
                schema that feedback will act on.
              </p>
            </div>
            <div className="actions">
              <button
                type="button"
                className="ghost"
                disabled={loading || busy}
                onClick={() =>
                  void loadGallery(
                    (offset + GALLERY_PAGE) % Math.max(galleryTotal - GALLERY_PAGE, 1),
                  )
                }
              >
                {loading ? "Loading…" : "Next page"}
              </button>
              <button
                type="button"
                className="primary"
                disabled={busy || baselineIndex === null}
                onClick={() => void startWithBaseline()}
              >
                {busy
                  ? "Starting…"
                  : baselineIndex === null
                    ? "Pick a baseline image"
                    : `Refine image #${baselineIndex} →`}
              </button>
            </div>
          </div>

          <div className="showcase-grid">
            {images.map((image) => {
              const isSelected = baselineIndex === image.index;
              return (
                <button
                  key={image.index}
                  type="button"
                  className={`candidate-card plain ${isSelected ? "selected" : ""}`}
                  onClick={() => setBaselineIndex(image.index)}
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

      {phase === "refine" && modify && (
        <section className="showcase-panel">
          <ModifyStage
            modify={modify}
            busy={busy}
            onFeedback={(text) =>
              void run(
                (id) => api.modifyFeedback(id, text),
                "Feedback parsed, hypotheses built, three probes sampled.",
              )
            }
            onSelectProbe={(probeId) =>
              void run(
                (id) => api.modifySelect(id, probeId),
                `Selected probe ${probeId}.`,
              )
            }
            onPreview={() =>
              void run((id) => api.modifyPreview(id), "Preview rendered (schema untouched).")
            }
            onCommit={() => void run((id) => api.modifyCommit(id), "Patch committed.")}
            onExecute={() => void run((id) => api.modifyExecute(id), "Patch executed.")}
            onVerify={() => void run((id) => api.modifyVerify(id), "Result verified.")}
            onContinue={() => {
              note(
                `Finished with ${modify.round_index} completed round(s). Use the studio to push the workflow to ComfyUI.`,
              );
              restart();
            }}
          />
        </section>
      )}

      {activity.length > 0 && (
        <section className="showcase-panel">
          <h2>Activity</h2>
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
