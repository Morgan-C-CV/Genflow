import { useCallback, useEffect, useMemo, useState } from "react";
import { ApiError, api } from "./api";
import CandidatesStage from "./components/CandidatesStage";
import ClarifyStage from "./components/ClarifyStage";
import ComposeStage from "./components/ComposeStage";
import RefineStage from "./components/RefineStage";
import SidePanel, { type TranscriptEntry } from "./components/SidePanel";
import Stepper, { type Stage } from "./components/Stepper";
import WorkflowStage from "./components/WorkflowStage";
import type {
  ComfyStatus,
  GeneratedImage,
  NormalizedSchema,
  PushResponse,
  RefinementState,
  RuntimeCandidate,
  RuntimePlan,
  RuntimeSession,
  RuntimeWall,
  WorkflowOptions,
  WorkflowResponse,
} from "./types";

/** Mirrors the CLI's exploitation loop length. */
const MAX_REFINE_ROUNDS = 8;
const REFINE_BATCH_SIZE = 6;

const DEFAULT_OPTIONS: WorkflowOptions = {
  width: 1024,
  height: 1024,
  batch_size: 1,
  seed: null,
  filename_prefix: "Genflow",
};

let entryCounter = 0;
function nextId(): string {
  entryCounter += 1;
  return `e${entryCounter}`;
}

function imageUrlFor(galleryIndex: number): string {
  return `/api/v1/runtime/gallery/image/${galleryIndex}`;
}

export default function App() {
  const [stage, setStage] = useState<Stage>("compose");
  const [session, setSession] = useState<RuntimeSession | null>(null);
  const [plan, setPlan] = useState<RuntimePlan | null>(null);
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);
  const [busy, setBusy] = useState(false);

  const [wall, setWall] = useState<RuntimeWall | null>(null);
  const [selectedIndices, setSelectedIndices] = useState<number[]>([]);
  const [refinement, setRefinement] = useState<RefinementState | null>(null);

  const [schema, setSchema] = useState<NormalizedSchema | null>(null);
  const [rawMetadata, setRawMetadata] = useState("");
  const [anchorSummary, setAnchorSummary] = useState("");

  const [options, setOptions] = useState<WorkflowOptions>(DEFAULT_OPTIONS);
  const [workflow, setWorkflow] = useState<WorkflowResponse | null>(null);
  const [push, setPush] = useState<PushResponse | null>(null);
  const [images, setImages] = useState<GeneratedImage[]>([]);
  const [polling, setPolling] = useState(false);

  const [comfyStatus, setComfyStatus] = useState<ComfyStatus | null>(null);

  const log = useCallback(
    (
      role: TranscriptEntry["role"],
      text: string,
      extra: Partial<TranscriptEntry> = {},
    ) => {
      setTranscript((previous) => [
        ...previous,
        { id: nextId(), role, text, ...extra },
      ]);
    },
    [],
  );

  const describeError = useCallback(
    (error: unknown, fallback: string): string => {
      if (error instanceof ApiError) return error.message;
      if (error instanceof Error) return error.message;
      return fallback;
    },
    [],
  );

  const refreshComfyStatus = useCallback(async () => {
    try {
      setComfyStatus(await api.comfyStatus());
    } catch (error) {
      setComfyStatus({
        reachable: false,
        base_url: "http://127.0.0.1:8188",
        comfyui_version: "",
        device: "",
        checkpoints: [],
        loras: [],
        samplers: [],
        error: describeError(error, "Could not read ComfyUI status."),
      });
    }
  }, [describeError]);

  useEffect(() => {
    void refreshComfyStatus();
  }, [refreshComfyStatus]);

  // Poll ComfyUI for rendered images once a prompt has been queued.
  useEffect(() => {
    if (!push?.pushed || !push.prompt_id) return;
    if (images.length > 0) return;

    let cancelled = false;
    setPolling(true);

    const tick = async () => {
      try {
        const result = await api.promptResult(push.prompt_id);
        if (cancelled) return;
        if (result.images.length > 0) {
          setImages(result.images);
          setPolling(false);
          log(
            "system",
            `ComfyUI produced ${result.images.length} image${
              result.images.length === 1 ? "" : "s"
            }.`,
          );
        } else if (result.status === "error") {
          setPolling(false);
          log("system", "ComfyUI execution failed. Check the ComfyUI console log.", {
            tone: "error",
          });
        }
      } catch {
        // Transient failures are expected while ComfyUI is still working.
      }
    };

    void tick();
    const timer = window.setInterval(tick, 3000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
      setPolling(false);
    };
  }, [push?.pushed, push?.prompt_id, images.length, log]);

  const completed = useMemo<Stage[]>(() => {
    const list: Stage[] = [];
    if (session) list.push("compose");
    if (session && plan && plan.next_action !== "ask_user") list.push("clarify");
    if (wall) list.push("candidates");
    if (session?.selected_gallery_index !== null && session?.selected_gallery_index !== undefined) {
      list.push("refine");
    }
    return list;
  }, [session, plan, wall]);

  /** The image the reference bundle was built from (for the workflow thumbnail). */
  const anchorCandidate = useMemo<RuntimeCandidate | null>(() => {
    const index = session?.selected_gallery_index;
    if (index === null || index === undefined) return null;
    const fromWall = wall?.candidates.find((item) => item.gallery_index === index);
    if (fromWall) return fromWall;
    return {
      slot: 0,
      gallery_index: index,
      id: "",
      image_url: imageUrlFor(index),
      local_path: "",
      prompt: "",
      negative_prompt: "",
      model: "",
      sampler: "",
      steps: "",
      cfgscale: "",
      seed: "",
      width: null,
      height: null,
      group_index: 0,
      group_label: "",
      distance: null,
    };
  }, [session?.selected_gallery_index, wall]);

  const loadCandidates = useCallback(
    async (sessionId: string, refresh: boolean) => {
      setBusy(true);
      if (refresh) log("user", "Shuffle images");
      try {
        const response = await api.candidates(sessionId, {
          refresh,
          per_query_k: 2,
          top_k: 16,
        });
        setSession(response.session);
        setPlan(response.plan);
        setWall(response.wall);
        setSelectedIndices([]);
        setStage("candidates");
        log(
          "agent",
          `Generated ${response.wall.candidates.length} images across ${response.wall.query_labels.length} retrieval directions.`,
          response.wall.description ? { detail: response.wall.description } : {},
        );
      } catch (error) {
        log("system", describeError(error, "Candidate generation failed."), {
          tone: "error",
        });
      } finally {
        setBusy(false);
      }
    },
    [describeError, log],
  );

  const handleStart = useCallback(
    async (intent: string) => {
      setBusy(true);
      setWall(null);
      setSelectedIndices([]);
      setRefinement(null);
      setSchema(null);
      setRawMetadata("");
      setWorkflow(null);
      setPush(null);
      setImages([]);
      log("user", intent);

      try {
        const response = await api.startEpisode(intent);
        setSession(response.session);
        setPlan(response.plan);
        log("agent", response.plan.reasoning_summary || "Intent analysis complete.");

        if (
          response.plan.next_action === "ask_user" &&
          response.plan.clarification_questions.length > 0
        ) {
          setStage("clarify");
          response.plan.clarification_questions.forEach((question) =>
            log("agent", question),
          );
        } else {
          await loadCandidates(response.session.session_id, false);
        }
      } catch (error) {
        log("system", describeError(error, "Failed to start the agent."), {
          tone: "error",
        });
      } finally {
        setBusy(false);
      }
    },
    [describeError, loadCandidates, log],
  );

  const handleClarify = useCallback(
    async (answers: string[]) => {
      if (!session) return;
      setBusy(true);
      const spoken = answers.filter((answer) => answer.trim());
      log("user", spoken.length > 0 ? spoken.join(" / ") : "(clarification skipped)");

      try {
        const response = await api.clarify(session.session_id, answers);
        setSession(response.session);
        setPlan(response.plan);

        const keepAsking =
          response.plan.next_action === "ask_user" &&
          response.plan.clarification_questions.length > 0 &&
          response.session.clarification_rounds < 4;

        if (keepAsking) {
          log("agent", "A few more things to confirm:");
          response.plan.clarification_questions.forEach((question) =>
            log("agent", question),
          );
        } else {
          log("agent", "Clarification closed. Retrieving candidate images.");
          await loadCandidates(response.session.session_id, false);
        }
      } catch (error) {
        log("system", describeError(error, "Failed to submit clarification."), {
          tone: "error",
        });
      } finally {
        setBusy(false);
      }
    },
    [session, describeError, loadCandidates, log],
  );

  const toggleCandidate = useCallback((candidate: RuntimeCandidate) => {
    setSelectedIndices((previous) =>
      previous.includes(candidate.gallery_index)
        ? previous.filter((index) => index !== candidate.gallery_index)
        : [...previous, candidate.gallery_index],
    );
  }, []);

  /** Shared tail: build the reference bundle result into a schema + result. */
  const runSchemaAndResult = useCallback(
    async (sessionId: string) => {
      log("agent", "Reference bundle ready. Generating metadata / schema…");
      const schemaResponse = await api.generateSchema(sessionId);
      setSession(schemaResponse.session);
      setSchema(schemaResponse.normalized);
      setRawMetadata(schemaResponse.raw_metadata);
      log("agent", "Schema generated.", {
        detail: schemaResponse.normalized.full_metadata_string || undefined,
      });

      try {
        const result = await api.produceResult(sessionId);
        setSession(result.session);
        log("agent", "Initial result produced. Ready to push to ComfyUI.");
      } catch (error) {
        log(
          "system",
          `Result production failed (workflow generation is unaffected): ${describeError(
            error,
            "",
          )}`,
          { tone: "muted" },
        );
      }
      setStage("workflow");
    },
    [describeError, log],
  );

  const handleUseDirectly = useCallback(async () => {
    if (!session || selectedIndices.length !== 1) return;
    setBusy(true);
    const galleryIndex = selectedIndices[0];
    log("user", `Use image #${galleryIndex} directly`);
    try {
      const selected = await api.select(session.session_id, galleryIndex);
      setSession(selected.session);
      setAnchorSummary(selected.anchor_summary);
      await runSchemaAndResult(session.session_id);
    } catch (error) {
      log("system", describeError(error, "Reference selection failed."), {
        tone: "error",
      });
    } finally {
      setBusy(false);
    }
  }, [session, selectedIndices, describeError, log, runSchemaAndResult]);

  const handleStartRefine = useCallback(async () => {
    if (!session || selectedIndices.length === 0) return;
    setBusy(true);
    log("user", `Refine seeds: ${selectedIndices.join(", ")}`);
    try {
      const response = await api.startRefinement(session.session_id, selectedIndices);
      setRefinement(response.refinement);
      setStage("refine");
      log(
        "agent",
        `Preference search seeded with ${response.refinement.seed_indices.length} image(s). Run a round to see candidates.`,
      );
    } catch (error) {
      log("system", describeError(error, "Could not start refinement."), {
        tone: "error",
      });
    } finally {
      setBusy(false);
    }
  }, [session, selectedIndices, describeError, log]);

  const handleRunRound = useCallback(async () => {
    if (!session) return;
    setBusy(true);
    try {
      const response = await api.refinementRound(session.session_id, REFINE_BATCH_SIZE);
      setRefinement(response.refinement);
      setSession(response.session);
      log(
        "agent",
        `Round ${response.refinement.round_index + 1}: ${response.refinement.pending_candidates.length} candidates. Mark the best and worst.`,
      );
    } catch (error) {
      log("system", describeError(error, "Could not run a refinement round."), {
        tone: "error",
      });
    } finally {
      setBusy(false);
    }
  }, [session, describeError, log]);

  const advanceAfterFeedback = useCallback(
    async (sessionId: string, roundIndex: number) => {
      if (roundIndex >= MAX_REFINE_ROUNDS) {
        log(
          "agent",
          `Reached ${MAX_REFINE_ROUNDS} rounds. Finish to use the best match.`,
        );
        return;
      }
      const next = await api.refinementRound(sessionId, REFINE_BATCH_SIZE);
      setRefinement(next.refinement);
      setSession(next.session);
    },
    [log],
  );

  const handleSubmitFeedback = useCallback(
    async (bestSlot: number, worstSlot: number) => {
      if (!session) return;
      setBusy(true);
      log("user", `Round feedback — best #${bestSlot}, worst #${worstSlot}`);
      try {
        const response = await api.refinementFeedback(session.session_id, {
          best_slot: bestSlot,
          worst_slot: worstSlot,
        });
        setRefinement(response.refinement);
        setSession(response.session);
        await advanceAfterFeedback(session.session_id, response.refinement.round_index);
      } catch (error) {
        log("system", describeError(error, "Could not submit feedback."), {
          tone: "error",
        });
      } finally {
        setBusy(false);
      }
    },
    [session, describeError, log, advanceAfterFeedback],
  );

  const handleSkipRound = useCallback(async () => {
    if (!session) return;
    setBusy(true);
    log("user", "Skip this round");
    try {
      const response = await api.refinementFeedback(session.session_id, { skip: true });
      setRefinement(response.refinement);
      setSession(response.session);
      await advanceAfterFeedback(session.session_id, response.refinement.round_index);
    } catch (error) {
      log("system", describeError(error, "Could not skip the round."), {
        tone: "error",
      });
    } finally {
      setBusy(false);
    }
  }, [session, describeError, log, advanceAfterFeedback]);

  const handleFinishRefine = useCallback(async () => {
    if (!session) return;
    setBusy(true);
    log("agent", "Fitting the preference model to pick the best match…");
    try {
      const response = await api.finishRefinement(session.session_id);
      setRefinement(response.refinement);
      setSession(response.session);
      setAnchorSummary(response.anchor_summary);
      log(
        "agent",
        `Best match selected (gallery index ${response.refinement.best_index}), reference bundle built from ${response.selected_reference_ids.length} images.`,
      );
      await runSchemaAndResult(session.session_id);
    } catch (error) {
      log("system", describeError(error, "Could not finish refinement."), {
        tone: "error",
      });
    } finally {
      setBusy(false);
    }
  }, [session, describeError, log, runSchemaAndResult]);

  const handleBuild = useCallback(async () => {
    if (!session) return;
    setBusy(true);
    setPush(null);
    try {
      const built = await api.buildWorkflow(session.session_id, options);
      setWorkflow(built);
      log(
        "agent",
        `Built the ComfyUI workflow (${Object.keys(built.api_graph).length} nodes).`,
      );
    } catch (error) {
      log("system", describeError(error, "Workflow build failed."), {
        tone: "error",
      });
    } finally {
      setBusy(false);
    }
  }, [session, options, describeError, log]);

  const handlePush = useCallback(async () => {
    if (!session) return;
    setBusy(true);
    setImages([]);
    try {
      const pushed = await api.pushWorkflow(session.session_id, options);
      setWorkflow(pushed);
      setPush(pushed);
      if (pushed.pushed) {
        log("agent", `Queued in ComfyUI, prompt_id ${pushed.prompt_id}.`);
      } else {
        log("system", pushed.error || "Push failed.", { tone: "error" });
      }
    } catch (error) {
      log("system", describeError(error, "Failed to push to ComfyUI."), {
        tone: "error",
      });
    } finally {
      setBusy(false);
    }
  }, [session, options, describeError, log]);

  const handleRefreshResult = useCallback(async () => {
    if (!push?.prompt_id) return;
    try {
      const result = await api.promptResult(push.prompt_id);
      if (result.images.length > 0) {
        setImages(result.images);
      } else {
        log("system", `No images yet (status: ${result.status ?? "pending"}).`, {
          tone: "muted",
        });
      }
    } catch (error) {
      log("system", describeError(error, "Failed to read the result."), {
        tone: "error",
      });
    }
  }, [push?.prompt_id, describeError, log]);

  const handleJump = useCallback(
    (target: Stage) => {
      if (target === "workflow" && !schema) return;
      if (target === "refine" && !refinement && !schema) return;
      if (target === "candidates" && !wall) return;
      if (target === "clarify" && !plan) return;
      setStage(target);
    },
    [schema, wall, plan, refinement],
  );

  const handleRefreshCandidates = useCallback(() => {
    if (session) void loadCandidates(session.session_id, true);
  }, [session, loadCandidates]);

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <div>
            <h1>Genflow Studio</h1>
            <p className="muted small">
              prompt → agent clarify → candidates → refine → ComfyUI workflow
            </p>
          </div>
        </div>
        {session && (
          <div className="session-info mono small">
            session {session.session_id.slice(0, 8)}
          </div>
        )}
      </header>

      <div className="app-body">
        <main className="app-main">
          <Stepper stage={stage} completed={completed} onJump={handleJump} />

          {stage === "compose" && (
            <ComposeStage busy={busy} onSubmit={handleStart} />
          )}

          {stage === "clarify" && plan && (
            <ClarifyStage
              plan={plan}
              busy={busy}
              onSubmit={handleClarify}
              onSkip={() => handleClarify([])}
            />
          )}

          {stage === "candidates" && wall && (
            <CandidatesStage
              wall={wall}
              busy={busy}
              selectedIndices={selectedIndices}
              onToggle={toggleCandidate}
              onRefresh={handleRefreshCandidates}
              onRefine={handleStartRefine}
              onUseDirectly={handleUseDirectly}
            />
          )}

          {stage === "refine" && refinement && (
            <RefineStage
              refinement={refinement}
              busy={busy}
              maxRounds={MAX_REFINE_ROUNDS}
              onRunRound={handleRunRound}
              onSubmitFeedback={handleSubmitFeedback}
              onSkipRound={handleSkipRound}
              onFinish={handleFinishRefine}
              onBackToCandidates={() => setStage("candidates")}
            />
          )}

          {stage === "workflow" && (
            <WorkflowStage
              schema={schema}
              rawMetadata={rawMetadata}
              anchorSummary={anchorSummary}
              selectedCandidate={anchorCandidate}
              options={options}
              onOptionsChange={setOptions}
              workflow={workflow}
              push={push}
              images={images}
              busy={busy}
              polling={polling}
              comfyStatus={comfyStatus}
              onBuild={handleBuild}
              onPush={handlePush}
              onRefreshResult={handleRefreshResult}
            />
          )}
        </main>

        <SidePanel
          transcript={transcript}
          plan={plan}
          status={comfyStatus}
          onRetryStatus={() => void refreshComfyStatus()}
        />
      </div>
    </div>
  );
}
