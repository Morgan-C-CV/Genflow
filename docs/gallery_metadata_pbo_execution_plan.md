# Gallery Metadata → Preference-Based Optimization: Execution Plan

## Objective

Build a first usable system that:

1. Parses the existing `metadata.json` into a normalized item table.
2. Extracts structured features only from the metadata already available.
3. Builds a metadata-based representation for each gallery item.
4. Creates coarse automatic classes for filtering and local search.
5. Supports user preference collection through pairwise or best/worst selection.
6. Trains a simple preference model first.
7. Runs local preference-based optimization before attempting global optimization.
8. Provides clear validation checks after each implementation step.

This plan is intentionally conservative. The priority is not model novelty. The priority is to produce a reliable, inspectable, debuggable first version.

---

# Phase 0 — Ground Rules

## Scope constraints

The system must only use information that already exists in the metadata. Do not add manual semantic labels. Do not depend on human annotation. Do not design around future fields that do not exist yet.

## Allowed information sources

Only use:

- raw metadata fields already present
- `prompt`
- `negative_prompt`
- any full metadata string if available
- image file path / url
- width / height / model / sampler / steps / cfg / seed / workflow / clipskip / ecosystem / dates if present
- LoRA syntax embedded in the prompt

## First-version modeling rule

Use:

- parsed metadata features
- simple coarse classification
- local candidate pools
- pairwise ranking first

Do not start with:

- end-to-end joint embedding learning
- global BO across the full dataset
- PCA as a default preprocessing step
- complex multimodal fusion as the first version

## Deliverables

The coding agent should produce:

- a metadata parser
- a normalized dataset table
- automatic coarse classification
- feature builder for metadata vectors
- preference feedback storage
- a simple pairwise preference model
- local candidate selection logic
- validation scripts and summary reports

---

# Phase 1 — Inspect and Normalize Metadata

## Goal

Create a unified schema from the raw `metadata.json`. The output must be a clean item table where each row corresponds to one gallery image.

## Required actions

### 1. Load the raw JSON safely

The code must:

- support list-of-dicts JSON
- support nested dict structures if present
- skip or log malformed records
- preserve original raw record for debugging

### 2. Define a normalized schema

Each item should contain these top-level fields where possible:

```json
{
  "item_id": "stable unique id",
  "image_url": "...",
  "local_path": "...",
  "prompt": "...",
  "negative_prompt": "...",
  "model": "...",
  "sampler": "...",
  "steps": 0,
  "cfgscale": 0.0,
  "seed": 0,
  "width": 0,
  "height": 0,
  "workflow": "...",
  "clipskip": null,
  "ecosystem": "...",
  "created_date": "...",
  "raw_record": {...}
}
```

### 3. Create a stable item ID

Priority order:

- use existing unique record id if present
- otherwise hash a combination of `local_path + image_url + prompt + seed`

### 4. Standardize missing values

Use a consistent policy:

- missing string → `""`
- missing numeric → `null`
- missing lists → `[]`

Do not silently convert missing to zero unless the field is truly numeric and zero is semantically valid.

## Required outputs

- `normalized_items.jsonl`
- `normalized_items.csv`
- parser log with counts:
  - total raw records
  - parsed records
  - skipped records
  - duplicate IDs
  - records missing prompt
  - records missing image path/url

## Validation

### Validation 1.1 — row count

Check that parsed row count matches expectation. If records were skipped, the reason must be logged.

### Validation 1.2 — uniqueness

Check `item_id` uniqueness. Fail if duplicates remain.

### Validation 1.3 — critical-field coverage

Report coverage percentage for:

- prompt
- negative\_prompt
- sampler
- steps
- cfgscale
- seed
- width
- height
- model
- workflow

### Validation 1.4 — schema integrity

Randomly sample 20 items and print normalized views to verify field consistency.

---

# Phase 2 — Parse Structured Signals from Prompt and Metadata

## Goal

Convert existing metadata into inspectable structured fields without manual labeling.

## Required actions

### 1. Parse LoRA references from prompt

Support patterns like:

```text
<lora:name:strength>
```

Extract:

- `lora_names`
- `lora_strengths`
- `lora_count`
- `avg_lora_strength`
- `max_lora_strength`

If parsing fails, keep empty lists and log the record.

### 2. Parse generation parameters from raw metadata strings if needed

If some fields are not already normalized but appear in free text, extract:

- steps
- cfg scale
- sampler
- scheduler / schedule type if present
- model
- clip skip
- denoising strength if present
- width / height if present

### 3. Build keyword parsers

Create rule-based parsers for these token groups:

#### Content keywords

Examples:

- portrait terms
- people terms
- landscape terms
- environment terms
- architecture/interior terms
- object/product terms
- animal terms
- vehicle/mecha terms

#### Style keywords

Examples:

- cinematic
- photorealistic
- analog film
- vintage
- anime
- illustration
- watercolor
- oil painting
- pixel art
- fantasy
- dark fantasy
- cyberpunk
- retro futurism

#### Shot/composition keywords

Examples:

- close-up
- full body
- side view
- wide shot
- top-down
- macro
- centered composition

#### Lighting keywords

Examples:

- rim light
- backlighting
- dramatic lighting
- soft light
- golden hour
- moody light
- low key
- volumetric light

#### Quality boilerplate keywords

Examples:

- masterpiece
- best quality
- highly detailed
- award-winning
- 8k
- ultra detailed

### 4. Store parsed fields

Each item should now include fields like:

```json
{
  "parsed": {
    "lora_names": [],
    "lora_strengths": [],
    "content_keywords": [],
    "style_keywords": [],
    "shot_keywords": [],
    "lighting_keywords": [],
    "quality_keywords": []
  }
}
```

## Required outputs

- `parsed_items.jsonl`
- `keyword_stats.json`
- top keyword frequency reports
- top LoRA frequency reports

## Validation

### Validation 2.1 — LoRA parsing accuracy

Randomly inspect 30 prompts that contain `<lora:` and verify parse correctness.

### Validation 2.2 — keyword parser sanity

Print top 50 content/style/lighting keywords. Check whether they are meaningful and not dominated by punctuation noise.

### Validation 2.3 — empty parse rate

Report percentage of items where each parsed category is empty. This should not be hidden. If a category is mostly empty, the parser rules need revision.

### Validation 2.4 — raw-to-parsed traceability

For 20 sampled items, show:

- original prompt
- extracted LoRAs
- extracted keyword groups This is mandatory for debugging.

---

# Phase 3 — Build Coarse Automatic Classes

## Goal

Create coarse classes for filtering and local candidate pool construction. These classes are not ground truth labels. They are operational filters.

## Required actions

### 1. Build `content_class`

Assign one primary content class by rule priority. Suggested label set:

- `portrait_character`
- `landscape_environment`
- `architecture_interior`
- `object_product`
- `animal_creature`
- `vehicle_machine`
- `anime_illustration`
- `other`

Use prompt keywords and parsed fields only.

### 2. Build `style_class`

Suggested label set:

- `photo_cinematic`
- `analog_vintage`
- `anime_illustration`
- `painterly_art`
- `fantasy_darkfantasy`
- `sci_fi_cyberpunk`
- `pixel_art`
- `other`

### 3. Build `pipeline_class`

Suggested label set:

- `txt2img_standard`
- `txt2img_lora`
- `fast_lcm_like`
- `unknown_pipeline`

This can be inferred from workflow names, LoRA presence, sampler, or obvious metadata patterns.

### 4. Build `model_family`

Use rule-based normalization. Examples:

- `sdxl_photo_family`
- `illustrious_pony_family`
- `cinematic_photo_family`
- `anime_family`
- `unknown_model_family`

## Required outputs

- `classified_items.jsonl`
- per-class counts
- confusion report for items matching multiple class rules

## Validation

### Validation 3.1 — class coverage

Every item must receive:

- one `content_class`
- one `style_class`
- one `pipeline_class`
- one `model_family`

If uncertain, assign `other` or `unknown_*`, never null.

### Validation 3.2 — ambiguous rule logging

Log records that matched multiple competing class rules before resolution.

### Validation 3.3 — spot-check by class

For each class, print 10 sampled items with prompt and assigned labels. Check obvious failure modes.

### Validation 3.4 — class balance

Report class distribution. If one class dominates almost everything, inspect parser bias.

---

# Phase 4 — Build Metadata Feature Representation

## Goal

Create a first usable metadata representation for preference modeling. This is the main first-version optimization space.

## Required actions

### 1. Build numeric features

Include where available:

- steps
- cfgscale
- width
- height
- aspect\_ratio
- clipskip
- lora\_count
- avg\_lora\_strength
- max\_lora\_strength
- prompt\_length
- negative\_prompt\_length
- count of style keywords
- count of content keywords
- count of lighting keywords

Apply standardization only to numeric columns.

### 2. Build categorical features

Encode:

- sampler
- workflow
- model\_family
- content\_class
- style\_class
- pipeline\_class

Use one-hot encoding for the first version. Avoid overengineering here.

### 3. Build text features

Use text embeddings for:

- prompt
- negative\_prompt
- concatenated LoRA names
- concatenated parsed keywords

Important rule: Keep these as separate embeddings first, then concatenate. Do not collapse everything into a single raw string.

### 4. Build final metadata vector

Construct:

$$
 z_{meta} = [z_{numeric}; z_{categorical}; z_{prompt}; z_{negative}; z_{lora}; z_{keywords}]
$$

Persist both:

- full feature dataframe
- final vector matrix

## Required outputs

- `feature_table.parquet`
- `z_meta.npy`
- `feature_spec.json`
- feature dimensionality report

## Validation

### Validation 4.1 — missing feature audit

Report missing rate per feature.

### Validation 4.2 — numeric scaling audit

Check mean/std after scaling for numeric columns.

### Validation 4.3 — categorical explosion audit

Report one-hot dimensionality per field. Flag extremely high-cardinality fields.

### Validation 4.4 — nearest-neighbor sanity check

Using only `z_meta`, retrieve nearest neighbors for 20 random items. Inspect whether neighbors are operationally similar:

- similar model family
- similar style class
- similar LoRA usage
- similar scene wording

This is one of the most important checks.

---

# Phase 5 — Build Image Embeddings as Secondary Signal

## Goal

Add image embeddings, but do not make them the primary optimization space in the first version.

## Required actions

### 1. Load images safely

Use `local_path` when available. Fallback to `image_url` only if needed. Log missing files.

### 2. Compute one fixed image embedding per item

Use one image encoder consistently. Do not compare encoders in the first implementation.

### 3. Store image vectors

Persist:

- `z_img.npy`
- image embedding dimensionality metadata

### 4. Keep image representation separate from metadata representation

Do not fuse into a single vector yet. The first version must preserve:

- `z_meta`
- `z_img`

## Required outputs

- `image_embedding_index.parquet` or mapping table
- `z_img.npy`
- image embedding coverage report

## Validation

### Validation 5.1 — image coverage

Report how many items successfully got an image embedding.

### Validation 5.2 — image neighbor sanity check

For 20 random items, inspect nearest neighbors under `z_img`. Check whether they are visually similar.

### Validation 5.3 — modality difference check

For the same anchor item, compare nearest neighbors under:

- `z_meta`
- `z_img`

This is required. It will show whether metadata and visual spaces behave differently.

---

# Phase 6 — Build Local Candidate Pool Logic

## Goal

Prevent the first preference model from operating over the entire heterogeneous dataset. Use local pools.

## Required actions

### 1. Define local pool construction rule

Recommended first rule:

- same `content_class`
- same `style_class`
- optionally same `model_family`

If the pool is too small, relax in this order:

1. drop `model_family`
2. keep `content_class`
3. keep `style_class`

### 2. Pool size rule

Target local pool size:

- minimum: 20
- preferred: 40–100
- maximum for first version: 200

### 3. Initial candidate diversification

When selecting initial images for the user:

- use metadata distance for broad spread
- optionally use image distance to prevent near-duplicates

## Required outputs

- candidate pool builder module
- pool statistics report

## Validation

### Validation 6.1 — pool coverage

Report how many items can form a valid local pool under the rule.

### Validation 6.2 — pool size distribution

Histogram of local pool sizes.

### Validation 6.3 — duplicate-like detection

For initial displayed candidates, check average pairwise similarity. Avoid showing nearly identical items.

---

# Phase 7 — Build Preference Feedback Storage

## Goal

Store user feedback in a way that can support pairwise modeling and future audit.

## Required actions

### 1. Create feedback schema

Support at least:

```json
{
  "feedback_id": "...",
  "session_id": "...",
  "user_id": "optional",
  "timestamp": "...",
  "interaction_type": "pairwise|best_worst|top1_from_k",
  "candidate_ids": ["..."],
  "preferred_id": "...",
  "rejected_id": "...",
  "best_id": "...",
  "worst_id": "...",
  "pool_context": {
    "content_class": "...",
    "style_class": "...",
    "model_family": "..."
  }
}
```

### 2. Convert best/worst to pairwise constraints

If user gives best/worst from 4 items, derive pairwise wins/losses. Store both the raw event and the derived pairwise edges.

### 3. Maintain pairwise training table

Create a table like:

```json
{
  "left_item_id": "...",
  "right_item_id": "...",
  "label": 1
}
```

where `label = 1` means left preferred over right.

## Required outputs

- feedback DB schema or tables
- pairwise edge builder
- event-to-edge conversion module

## Validation

### Validation 7.1 — referential integrity

Every feedback item ID must exist in the gallery item table.

### Validation 7.2 — conversion correctness

For sampled best/worst events, verify generated pairwise edges manually.

### Validation 7.3 — duplicate-edge policy

Define whether repeated comparisons are stored separately or aggregated. Validate consistency.

---

# Phase 8 — Train a Simple Preference Model First

## Goal

Train a first preference model before implementing GP-based PBO. This model should be simple, inspectable, and good enough to validate the interaction loop.

## Required actions

### 1. Use pairwise ranking first

Recommended first model:

- Bradley-Terry style pairwise logistic model or
- pairwise logistic regression on feature differences

For an item pair `(i, j)`, train on:

$$
 x_{ij} = z_{meta}(i) - z_{meta}(j)
$$

with label:

- 1 if `i` preferred over `j`
- 0 otherwise

### 2. Produce item utility scores

After training, be able to output a score per item within a local pool.

### 3. Expose model confidence proxy if possible

Even if not a full GP uncertainty, provide at least:

- margin score
- calibrated probability
- ensemble variance if implemented later

## Required outputs

- train/eval script
- trained pairwise model artifact
- item scoring API/module

## Validation

### Validation 8.1 — offline pairwise accuracy

Train/validation split over pairwise data. Report pairwise prediction accuracy.

### Validation 8.2 — ranking sanity in local pools

For several pools, print top-ranked items and inspect whether they align with the observed feedback.

### Validation 8.3 — stability under retraining

Retrain with different random seeds and check whether ranking changes are reasonable, not chaotic.

---

# Phase 9 — Implement First Local Preference-Based Optimization Loop

## Goal

Close the loop: show candidates, collect preference, update model, select next candidates.

## Required actions

### 1. Start from a local pool

Do not optimize globally. A session begins inside one pool.

### 2. Select initial candidates

First round can use:

- metadata diversity sampling
- class-consistent candidates

### 3. Update preference model after each round

After each user choice:

- append pairwise edges
- retrain or incrementally update model
- rescore pool

### 4. Select next comparison candidates

First version rule:

- keep current best item as champion
- choose one challenger with high predicted utility and high uncertainty proxy
- optionally choose 2 challengers if UI supports 3-way or 4-way display

### 5. Stop condition

Stop after one of:

- fixed interaction budget reached
- predicted top item stable for N rounds
- user explicitly accepts current item

## Required outputs

- session loop implementation
- next-candidate selection module
- interaction log

## Validation

### Validation 9.1 — loop integrity

Run a simulated session end-to-end without UI using synthetic preferences. Ensure no crashes.

### Validation 9.2 — convergence behavior

In simulation, check whether the loop tends to recover the planted preferred region.

### Validation 9.3 — repeated-item rate

Measure how often the same items are repeatedly shown. This must be monitored.

### Validation 9.4 — candidate diversity

Even in local BO, challenger items should not all be near-duplicates of the current best.

---

# Phase 10 — Add Image Embedding as Secondary Reranking or Diversity Signal

## Goal

Only after the metadata-first loop works, add image embeddings carefully.

## Required actions

### Option A — diversity constraint

Use `z_img` to avoid showing visually near-identical challengers.

### Option B — residual reranking

Within top metadata candidates, rerank by visual diversity or a small image-based similarity penalty.

### Option C — dual-kernel baseline

Add a baseline scoring or similarity scheme using:

$$
 k(i,j) = \alpha k_{meta}(i,j) + \beta k_{img}(i,j)
$$

Use fixed weights first. Do not optimize weights before the baseline exists.

## Validation

### Validation 10.1 — duplicate suppression

Check whether image-aware reranking reduces near-duplicate display frequency.

### Validation 10.2 — preference impact

Compare with metadata-only loop:

- number of rounds to satisfactory item
- repeated display rate
- local ranking coherence

---

# Phase 11 — PCA Policy

## Goal

Define clearly whether PCA is used.

## Required rule

PCA must **not** be the default preprocessing step before the first preference model.

## Allowed uses of PCA

### Use 1 — visualization

Use PCA to inspect the metadata space visually. Examples:

- plot by `content_class`
- plot by `style_class`
- highlight selected/preferred items

### Use 2 — ablation

Compare:

- metadata only
- metadata + PCA(32)
- metadata + PCA(16)

### Use 3 — local compression only if necessary

Only if feature dimensionality becomes unstable for the chosen model, test PCA on local pools.

## Validation

### Validation 11.1 — PCA variance report

If PCA is used, report explained variance ratio.

### Validation 11.2 — utility loss check

Compare nearest-neighbor behavior before and after PCA. If PCA destroys operational neighbors, reject it.

### Validation 11.3 — model comparison

Compare pairwise ranking performance with and without PCA. Do not keep PCA just because it is mathematically convenient.

---

# Phase 12 — Minimum Experiment Matrix

## Goal

Produce a first comparison table, not just a single pipeline.

## Required experiments

### Experiment A

Metadata-only local preference model

### Experiment B

Metadata-only + PCA local preference model

### Experiment C

Image-only local preference model

### Experiment D

Metadata-first with image-based diversity reranking

### Experiment E

Optional fixed weighted fusion baseline

## Required metrics

- pairwise prediction accuracy
- rounds to stable top item
- repeated-item rate
- average visual similarity among displayed candidates
- local pool size
- coverage of successful sessions

## Validation

### Validation 12.1 — same evaluation protocol

All experiments must use the same session budget and same local-pool rules.

### Validation 12.2 — ablation clarity

Each experiment must change only one major factor at a time.

---

# Final Engineering Requirements Summary

## What the coding agent must build first

1. metadata normalizer
2. prompt/LoRA parser
3. coarse classifier
4. metadata feature builder
5. image feature extractor
6. local pool constructor
7. preference feedback storage
8. pairwise ranking model
9. local optimization loop
10. validation scripts for every stage

## What the coding agent must not do first

1. do not start with global BO
2. do not start with end-to-end multimodal training
3. do not make PCA mandatory
4. do not hide parser failures
5. do not silently drop malformed records without logs
6. do not fuse metadata and image embeddings too early

---

# Acceptance Checklist

The first version is complete only if all of the following are true:

- metadata can be normalized reproducibly
- LoRA and keyword parsing are traceable
- every item gets coarse operational classes
- metadata vectors can retrieve operationally similar neighbors
- local pools are valid and not trivially tiny
- preference events can be stored and converted to pairwise edges
- the simple pairwise model can rank within local pools
- the loop can run end-to-end on simulated or real feedback
- each phase has a validation report
- PCA is treated as optional analysis, not mandatory preprocessing

---

# Suggested Directory Structure

```text
project/
  data/
    raw/
    processed/
  src/
    parsing/
    normalization/
    classification/
    features/
    embeddings/
    pools/
    feedback/
    preference_model/
    optimization_loop/
    validation/
  reports/
  notebooks/
  configs/
```

---

# Suggested Immediate Next Action

The coding agent should start with Phase 1 and Phase 2 only. Do not implement the preference model before the parsed metadata outputs pass validation.

