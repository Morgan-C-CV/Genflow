# Genflow Studio — Frontend

An interactive frontend for the Genflow agent covering the full pipeline:

```
User enters a prompt
   ↓
Genflow agent plans the intent (may ask clarifying questions over several rounds)
   ↓
Retrieves a divergent wall of 16 candidate images (shufflable, never repeating)
   ↓
User picks the reference image → schema + initial result
   ↓
Refine: feedback → LLM reads the axes → three gallery references per axis
(near / mid / far) → you pick one per axis → LLM composes one schema → commit
   ↓
Converts the schema into ComfyUI workflow JSON → pushes it to the ComfyUI queue
```

## Quick start

Three processes are involved: **ComfyUI**, the **Genflow backend**, and **this frontend**.

```bash
# 1) ComfyUI (the minimal install inside this repo)
cd ../../ComfyUI && ./start.sh          # http://127.0.0.1:8188

# 2) Genflow backend
cd ../Genflow/backend && ./start.sh      # http://127.0.0.1:8000

# 3) Frontend
cd ../frontend && ./start.sh             # http://127.0.0.1:5173
```

Then open **http://127.0.0.1:5173**.

The Vite dev server proxies `/api` to `127.0.0.1:8000`, so no extra CORS setup is
needed. If the backend runs elsewhere, set `VITE_API_BASE`
(e.g. `VITE_API_BASE=http://10.0.0.5:8000/api/v1`).

## Routes

| Route | Page |
|---|---|
| `/` | The studio — the full prompt → clarify → candidates → refine → workflow flow |
| `/showcase/refine` | The same studio, seeded with a start point and opened on the Refine stage |

There is no routing dependency: `main.tsx` inspects
`window.location.pathname` at startup and passes a flag to `App`. Vite's dev
server falls back to `index.html`, so a hard refresh on `/showcase/refine` works.

### `/showcase/refine`

Renders the same studio app as `/`; nothing about the interface changes. Refine
acts on an existing result, so this entry point supplies one — a gallery record's
prompt, sampler and model become the current schema — and opens the Refine stage
directly. Everything from the feedback box onward is the ordinary shift/modify
loop.

Pick a specific record with `?start=<gallery index>`; otherwise one is chosen at
random so repeated visits start from different points.

The seeding endpoint is planner-free by design (there is no intent to plan, since
the start point is given), so this route makes no LLM calls. Execution inside the
loop still goes through the repo's mock `ResultExecutor`; push the workflow from
the studio to actually render.

## LLM configuration

The backend uses the **DeepSeek official endpoint** by default
(model `deepseek-flash`, i.e. DeepSeek-V4.1-Flash).

- The key is read from `Genflow/.env_ds` (accepts `api_key` or `DEEPSEEK_API_KEY`).
  That file is git-ignored.
- Related environment variables: `LLM_PROVIDER` (`deepseek` / `gemini`),
  `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL`, `LLM_REQUEST_TIMEOUT`.
- When `LLM_PROVIDER` is unset the backend auto-selects: DeepSeek if a key is
  present, otherwise Gemini (which requires `GOOGLE_API_KEY`).

## Layout

| Area | Purpose |
|---|---|
| Step bar (top) | Intent → Clarify → Candidates → Refine → Workflow; completed steps are clickable |
| Main column (left) | Interface for the current stage |
| Side rail (right) | ComfyUI connection status, agent plan (locked/open axes, fixed constraints), full conversation transcript |

The **Candidates** stage shows images only — no per-image metadata — grouped by
retrieval direction. Picking one image is the whole selection: it becomes the
reference the schema is composed from.

The **Refine** stage works on an existing result:

1. You describe what is wrong and what must stay. An **interpretation model** reads
   it and splits it into modification axes. This is semantic, not keyword matching,
   so "flat and washed out" resolves to `lighting_vibe` and `color_palette` even
   though it names neither. If the model is unavailable the keyword parser is the
   fallback, and the UI shows which one was used.
2. For **each axis**, a retrieval query from that reading is projected into the
   PBO space and three **real gallery images** are returned at increasing distance
   from the current result along that axis: `near`, `mid`, `far`. The PBO scorer
   pre-picks one per axis; you can override any of them.
3. Once you have picked one reference per axis, the **composition model** unifies
   them into a single schema. Preview shows that proposal while the committed
   schema stays exactly as it was.
4. Commit is what applies it, then execute and verify close the round.

The loop is capped at three rounds, matching the user study.

The workflow stage offers two JSON views:

- **API workflow JSON** — the `POST /prompt` payload; usable directly with `curl`.
- **Canvas workflow JSON** — ComfyUI's canvas format (`nodes`/`links`), which can be
  dragged into the ComfyUI UI for further editing.

Both support copy and download.

## API endpoints used

Routes are defined in `backend/src/app/api/v1/endpoints/runtime.py`
(prefix `/api/v1/runtime`).

| Method | Path | Purpose |
|---|---|---|
| POST | `/episodes` | Create a session, return the plan |
| POST | `/episodes/{id}/clarify` | Submit answers (`answers: []` declines clarification) |
| POST | `/episodes/{id}/candidates` | Generate the candidate wall |
| POST | `/episodes/{id}/select` | Pick a candidate and build the reference bundle |
| POST | `/episodes/{id}/modify/feedback` | Read the axes, retrieve 3 gallery references per axis |
| POST | `/episodes/{id}/modify/select` | Choose one reference for one axis |
| POST | `/episodes/{id}/modify/preview` | Compose a schema without touching the committed one |
| POST | `/episodes/{id}/modify/commit` | Apply the composed schema |
| POST | `/episodes/{id}/modify/execute` | Σ5→Σ6: run the committed patch |
| POST | `/episodes/{id}/modify/verify` | Σ6→Σ7: verify, then close the round |
| GET | `/episodes/{id}/modify` | Current Σ state |
| POST | `/episodes/{id}/schema` | Generate metadata / schema |
| POST | `/episodes/{id}/result` | Produce the initial result |
| POST | `/episodes/{id}/workflow` | Build the workflow (without pushing) |
| POST | `/episodes/{id}/workflow/push` | Build and push to the ComfyUI queue |
| GET | `/workflow/result/{prompt_id}` | Poll for generated images |
| GET | `/comfyui/status` | ComfyUI reachability and available assets |
| GET | `/gallery/images` | Gallery index listing (served without warming the embedding stack) |
| GET | `/gallery/image/{index}?w=` | Original image, or a cached downscaled thumbnail |
| POST | `/showcase/episode` | Planner-free session over hand-picked gallery images (used by tooling only) |
| POST | `/showcase/modify` | Seeds a current result from a gallery record, for `/showcase/refine` |

The `/refine/*` PBO endpoints from an earlier iteration are still served for
`backend/src/tests/cli_test.py`; the UI no longer uses them.

`/showcase/episode` is not used by any page — only by tooling. `/showcase/modify`
is what `/showcase/refine` calls to obtain its start point.

Gallery originals average ~1.8 MB (some exceed 20 MB at 2560×3712), so grids
request `?w=640` thumbnails. Those are generated once with Pillow and cached in
`Genflow/backend/.thumbnails/` — a 24 MB original becomes ~39 KB and is served
in ~2 ms once cached.

## Two things to know about the ComfyUI integration

1. **A checkpoint is required.** The schema's `model` is a Civitai display name
   (e.g. `Juggernaut_XL_-_Ragnarok_by_RunDiffusion`) while ComfyUI needs an on-disk
   filename. Genflow fuzzy-matches them; when `checkpoint_resolved=false` the push
   button is **disabled** and the reason is shown. Drop the model into
   `ComfyUI/models/checkpoints/` and the button becomes available once it matches.

2. **`<lora:...>` tags are stripped.** The schema's prompt may contain A1111-style
   `<lora:name:weight>` tags, which core ComfyUI does not parse. Genflow removes them
   from the text, tries to match each name against the real files in
   `ComfyUI/models/loras/`, inserts a `LoraLoader` node for every hit, and lists the
   misses under warnings.

Samplers are mapped from A1111 naming to ComfyUI (e.g. `DPM++ 2M Karras` →
`dpmpp_2m` + `karras`); unsupported values fall back with a warning.

## Verification

The backend ships an end-to-end smoke script covering the clarification dialogue,
candidates, schema, workflow structure, and the push:

```bash
cd backend/src && ../.venv/bin/python e2e_runtime_api.py
```

It runs two scenarios: a vague intent (triggers clarification) and a specific intent
(goes straight to retrieval).

## Known limitations

- **No checkpoint means no image.** The workflow JSON is still generated correctly,
  but the push is refused — by design.
- **Conversation rounds:** the frontend stops after 4 clarification rounds and
  continues; the backend itself has no cap.
- **Refinement rounds:** the modify stage is capped at three rounds. Within a
  round, PBO pre-picks one gallery reference per axis and you can change any of
  them. References are shown `near` → `mid` → `far`, which is increasing distance
  from the current result; the PBO score is a separate ranking shown on each card.
- **Refine spends LLM calls:** one to read the feedback, one to compose the schema
  per preview. Both fall back to deterministic rules if the model is unavailable,
  which the UI reports as "read by rules" / a composition source of `rules`.
- **Sessions are stored in memory.** Restarting the backend loses them; start over.
- Generation parameters (width/height/batch/seed) come from the frontend, since the
  schema carries no resolution field. Default is 1024×1024.
- Agent-authored text (plan reasoning, retrieval labels) is produced by the LLM and
  follows the language of your prompt; the UI chrome itself is English only.
