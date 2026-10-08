# Genflow Studio — Frontend

An interactive frontend for the Genflow agent covering the full pipeline:

```
User enters a prompt
   ↓
Genflow agent plans the intent (may ask clarifying questions over several rounds)
   ↓
Retrieves a divergent wall of 16 candidate images (shufflable, never repeating)
   ↓
User selects one or more seed images
   ↓
Optional: preference search (PBO) — rounds of 6 images, marked best/worst,
each round steering the next, until the model converges on the best match
   ↓
Generates metadata / schema (prompt, model, sampler, steps, cfg, seed, lora…)
   ↓
Converts it into ComfyUI workflow JSON → pushes it to the ComfyUI queue
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
| `/showcase/refine` | Interactive refine walkthrough over real gallery images |

Routing is a ~40-line history-API helper (`src/router.tsx`) rather than a router
dependency, since the app has two entry points. Vite's dev server already falls
back to `index.html`, so a hard refresh on `/showcase/refine` works.

### `/showcase/refine`

A self-contained walkthrough of the preference search: pick seed images from the
gallery, then run rounds of six candidates, marking **Best** and **Worst**, and
finish to see the model's chosen match. It drives the **real** refinement
endpoints and the real Gaussian-process fit, but skips the planner and retrieval
pipeline — the seed wall is whatever you pick — so it costs no LLM calls.

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
retrieval direction. Click images to toggle them as seeds, then either run the
preference search or use a single image directly.

The **Refine** stage runs the PBO loop: each round proposes 6 images (two exploit
the current best, three explore nearby, one explores a distant region). Mark one
as **Best** and another as **Worst** to steer the next round, or skip a round
(which widens the search). Finishing fits a Gaussian process over all feedback and
uses its argmax as the final reference.

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
| POST | `/episodes/{id}/refine/start` | Seed the PBO refinement loop (one or more indices) |
| POST | `/episodes/{id}/refine/round` | Run one round, returning 6 candidates |
| POST | `/episodes/{id}/refine/feedback` | Mark best/worst slots, or `skip: true` |
| POST | `/episodes/{id}/refine/finish` | Fit the preference model and build the reference bundle |
| GET | `/episodes/{id}/refine` | Current refinement state |
| POST | `/episodes/{id}/schema` | Generate metadata / schema |
| POST | `/episodes/{id}/result` | Produce the initial result |
| POST | `/episodes/{id}/workflow` | Build the workflow (without pushing) |
| POST | `/episodes/{id}/workflow/push` | Build and push to the ComfyUI queue |
| GET | `/workflow/result/{prompt_id}` | Poll for generated images |
| GET | `/comfyui/status` | ComfyUI reachability and available assets |
| GET | `/gallery/images` | Gallery index listing (served without warming the embedding stack) |
| GET | `/gallery/image/{index}?w=` | Original image, or a cached downscaled thumbnail |
| POST | `/showcase/episode` | Planner-free session over hand-picked gallery images |

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
- **Refinement rounds:** the frontend targets 8 rounds (matching `cli_test.py`) but
  you can finish after any completed round. Within a round the exploit slots
  deliberately repeat images already rated highly; only the exploration slots bring
  new ones, so later rounds mix confirmed picks with fresh candidates.
- **Sessions are stored in memory.** Restarting the backend loses them; start over.
- Generation parameters (width/height/batch/seed) come from the frontend, since the
  schema carries no resolution field. Default is 1024×1024.
- Agent-authored text (plan reasoning, retrieval labels) is produced by the LLM and
  follows the language of your prompt; the UI chrome itself is English only.
