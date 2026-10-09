import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, api } from "./api";
import CandidatesStage from "./components/CandidatesStage";
import ClarifyStage from "./components/ClarifyStage";
import ComposeStage from "./components/ComposeStage";
import ModifyStage from "./components/ModifyStage";
import SidePanel, { type TranscriptEntry } from "./components/SidePanel";
import Stepper, { type Stage } from "./components/Stepper";
import WorkflowStage from "./components/WorkflowStage";
import type {
  ComfyStatus,
  GeneratedImage,
  ModifyState,
  NormalizedSchema,
  PushResponse,
  RuntimeCandidate,
  RuntimePlan,
  RuntimeSession,
  RuntimeWall,
  WorkflowOptions,
  WorkflowResponse,
} from "./types";

const DEFAULT_OPTIONS: WorkflowOptions = {
  width: 1024,
  height: 1024,
  batch_size: 1,
  seed: null,
  filename_prefix: "Genflow",
  checkpoint_override: null,
};

let entryCounter = 0;
function nextId(): string {
  entryCounter += 1;
  return `e${entryCounter}`;
}

function imageUrlFor(galleryIndex: number): string {
  return `/api/v1/gallery/image/${galleryIndex}?w=768`;
}

interface AppProps {
  /**
   * Refine operates on an existing result, so the refine entry point seeds one
   * from a gallery record and opens straight into the modify stage. The rest of
   * the app is unchanged.
   */
  seedRefineFromGallery?: boolean;
}

export default function App({ seedRefineFromGallery = false }: AppProps) {
  const [stage, setStage] = useState<Stage>("compose");
  const [session, setSession] = useState<RuntimeSession | null>(null);
  const [plan, setPlan] = useState<RuntimePlan | null>(null);
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);
  const [busy, setBusy] = useState(false);

  const [wall, setWall] = useState<RuntimeWall | null>(null);
  const [selectedIndices, setSelectedIndices] = useState<number[]>([]);
  const [modify, setModify] = useState<ModifyState | null>(null);

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

  // /showcase/refine: refine acts on an existing result, so seed one from a
  // gallery record (override with ?start=<index>) and open the modify stage.
  // StrictMode invokes effects twice, so the request is created once and both
  // invocations subscribe to the same promise; the backend sees a single call.
  const seedPromiseRef = useRef<Promise<{
    session: RuntimeSession;
    modify: ModifyState;
    plan: RuntimePlan;
  }> | null>(null);

  useEffect(() => {
    if (!seedRefineFromGallery) return;

    if (!seedPromiseRef.current) {
      seedPromiseRef.current = (async () => {
        const requested = Number.parseInt(
          new URLSearchParams(window.location.search).get("start") ?? "",
          10,
        );
        let index: number | null = Number.isFinite(requested) ? requested : null;
        if (index === null) {
          const { total } = await api.galleryCount();
          index = total > 0 ? Math.floor(Math.random() * total) : null;
        }

        const seeded = await api.startRefineFromGallery(index);
        const episode = await api.episode(seeded.session.session_id);
        return {
          session: seeded.session,
          modify: seeded.modify,
          plan: episode.plan,
        };
      })();
    }

    let active = true;
    seedPromiseRef.current
      .then((data) => {
        if (!active) return;
        setSession(data.session);
        setModify(data.modify);
        setPlan(data.plan);
        setStage("modify");
      })
      .catch((error) => {
        if (!active) return;
        log("system", describeError(error, "Could not open the refine loop."), {
          tone: "error",
        });
      });

    return () => {
      active = false;
    };
  }, [seedRefineFromGallery, describeError, log]);

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
    if (modify && modify.stage === "verified") list.push("modify");
    return list;
  }, [session, plan, wall, modify]);

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
      setModify(null);
      setSchema(null);
      setRawMetadata("");
      setWorkflow(null);
      setPush(null);
      setImages([]);
      setOptions(DEFAULT_OPTIONS);
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
    setSelectedIndices([candidate.gallery_index]);
  }, []);

  /** Create stage tail: build the reference result into a schema + result, then refine. */
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
        log("agent", "Initial result produced. You can now refine it.");
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

      // Open the modify state so the feedback box is ready immediately.
      try {
        const modifyState = await api.modifyState(sessionId);
        setModify(modifyState.modify);
      } catch (error) {
        log("system", describeError(error, "Could not open the refine loop."), {
          tone: "muted",
        });
      }
      setStage("modify");
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

  /* ---------- shift/modify loop (thesis 4.3) ---------- */

  const runModify = useCallback(
    async (
      action: (sessionId: string) => Promise<{ modify: ModifyState }>,
      label: string,
    ) => {
      if (!session) return;
      setBusy(true);
      try {
        const response = await action(session.session_id);
        setModify(response.modify);
        log("agent", label);
      } catch (error) {
        log("system", describeError(error, "The refinement step failed."), {
          tone: "error",
        });
      } finally {
        setBusy(false);
      }
    },
    [session, describeError, log],
  );

  const handleModifyFeedback = useCallback(
    async (text: string) => {
      if (!session) return;
      setBusy(true);
      log("user", text);
      try {
        const response = await api.modifyFeedback(session.session_id, text);
        setModify(response.modify);
        const axes = response.modify.dissatisfaction_axes.join(", ") || "none";
        log(
          "agent",
          `Parsed axes: ${axes}. Built ${response.modify.hypotheses.length} hypotheses and sampled ${response.modify.probes.length} probes.`,
        );
      } catch (error) {
        log("system", describeError(error, "Feedback analysis failed."), {
          tone: "error",
        });
      } finally {
        setBusy(false);
      }
    },
    [session, describeError, log],
  );

  const handleSelectProbe = useCallback(
    (probeId: string) =>
      void runModify(
        (id) => api.modifySelect(id, probeId),
        `Selected probe ${probeId}.`,
      ),
    [runModify],
  );

  const handleModifyPreview = useCallback(
    () =>
      void runModify(
        (id) => api.modifyPreview(id),
        "Preview rendered. The committed schema is untouched.",
      ),
    [runModify],
  );

  const handleModifyCommit = useCallback(
    () => void runModify((id) => api.modifyCommit(id), "Patch committed to the schema."),
    [runModify],
  );

  const handleModifyExecute = useCallback(
    () =>
      void runModify(
        (id) => api.modifyExecute(id),
        "Executed the committed patch.",
      ),
    [runModify],
  );

  const handleModifyVerify = useCallback(async () => {
    if (!session) return;
    setBusy(true);
    try {
      const response = await api.modifyVerify(session.session_id);
      setModify(response.modify);
      const verifier = response.modify.verifier ?? {};
      log(
        "agent",
        `Verified round ${response.modify.round_index}: improved=${verifier.improved}, confidence=${verifier.confidence}, continue=${response.modify.continue_recommended}.`,
      );
    } catch (error) {
      log("system", describeError(error, "Verification failed."), { tone: "error" });
    } finally {
      setBusy(false);
    }
  }, [session, describeError, log]);

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

  const handleApplyCheckpoint = useCallback(
    async (checkpoint: string) => {
      if (!session) return;
      const next: WorkflowOptions = {
        ...options,
        checkpoint_override: checkpoint,
      };
      setOptions(next);
      setBusy(true);
      setImages([]);
      log("user", `Use checkpoint ${checkpoint}`);
      try {
        const pushed = await api.pushWorkflow(session.session_id, next);
        setWorkflow(pushed);
        setPush(pushed);
        if (pushed.pushed) {
          log(
            "agent",
            `Queued in ComfyUI with checkpoint ${checkpoint}, prompt_id ${pushed.prompt_id}.`,
          );
        } else {
          log("system", pushed.error || "Push failed.", { tone: "error" });
        }
      } catch (error) {
        log(
          "system",
          describeError(error, "Failed to push with the chosen checkpoint."),
          { tone: "error" },
        );
      } finally {
        setBusy(false);
      }
    },
    [session, options, describeError, log],
  );

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
      if (target === "modify" && !modify && !schema) return;
      if (target === "candidates" && !wall) return;
      if (target === "clarify" && !plan) return;
      setStage(target);
    },
    [schema, wall, plan, modify],
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
              selectedIndex={selectedIndices[0] ?? null}
              onSelect={toggleCandidate}
              onRefresh={handleRefreshCandidates}
              onConfirm={handleUseDirectly}
            />
          )}

          {stage === "modify" && modify && (
            <ModifyStage
              modify={modify}
              busy={busy}
              onFeedback={(text) => void handleModifyFeedback(text)}
              onSelectProbe={handleSelectProbe}
              onPreview={handleModifyPreview}
              onCommit={handleModifyCommit}
              onExecute={handleModifyExecute}
              onVerify={() => void handleModifyVerify()}
              onContinue={() => setStage("workflow")}
            />
          )}

          {stage === "modify" && !modify && schema && (
            <div className="stage">
              <h1>Refine the current result</h1>
              <p className="lede">
                A result is ready. Describe what you would change about it to start the
                refinement loop.
              </p>
              <div className="compose-actions">
                <button
                  type="button"
                  className="primary"
                  disabled={busy}
                  onClick={() => {
                    if (!session) return;
                    setBusy(true);
                    api
                      .modifyState(session.session_id)
                      .then((response) => setModify(response.modify))
                      .catch((error) =>
                        log("system", describeError(error, "Could not load the refine state."), {
                          tone: "error",
                        }),
                      )
                      .finally(() => setBusy(false));
                  }}
                >
                  Start refining
                </button>
                <button
                  type="button"
                  className="ghost"
                  onClick={() => setStage("workflow")}
                >
                  Skip to workflow
                </button>
              </div>
            </div>
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
              onApplyCheckpoint={handleApplyCheckpoint}
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
