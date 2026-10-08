import type { RuntimeCandidate, RuntimeWall } from "../types";

interface CandidatesStageProps {
  wall: RuntimeWall;
  busy: boolean;
  selectedIndices: number[];
  onToggle: (candidate: RuntimeCandidate) => void;
  onRefresh: () => void;
  onRefine: () => void;
  onUseDirectly: () => void;
}

export default function CandidatesStage({
  wall,
  busy,
  selectedIndices,
  onToggle,
  onRefresh,
  onRefine,
  onUseDirectly,
}: CandidatesStageProps) {
  const selected = new Set(selectedIndices);
  const selectedCount = selectedIndices.length;

  // Preserve the group ordering the agent returned, one row per direction.
  const groups: { label: string; items: RuntimeCandidate[] }[] = [];
  for (const candidate of wall.candidates) {
    const label =
      candidate.group_label || wall.query_labels[candidate.group_index - 1] || "";
    const last = groups[groups.length - 1];
    if (last && last.label === label) last.items.push(candidate);
    else groups.push({ label, items: [candidate] });
  }

  return (
    <div className="stage candidates-stage">
      <div className="stage-head">
        <div>
          <h1>Pick seed images</h1>
          <p className="lede">
            {wall.candidates.length} images across {groups.length} retrieval
            directions. Select one or more seeds, then refine them with a
            preference search or use a single one directly.
          </p>
        </div>
        <button type="button" className="ghost" onClick={onRefresh} disabled={busy}>
          {busy ? "Generating…" : "Shuffle images"}
        </button>
      </div>

      <div className="selection-bar">
        <span className="selection-count">
          {selectedCount === 0
            ? "No images selected"
            : `${selectedCount} image${selectedCount === 1 ? "" : "s"} selected`}
        </span>
        <div className="actions">
          <button
            type="button"
            className="ghost"
            onClick={onUseDirectly}
            disabled={busy || selectedCount !== 1}
            title={
              selectedCount === 1
                ? undefined
                : "Select exactly one image to use it directly"
            }
          >
            Use selected image directly
          </button>
          <button
            type="button"
            className="primary"
            onClick={onRefine}
            disabled={busy || selectedCount === 0}
            title={selectedCount === 0 ? "Select at least one image" : undefined}
          >
            {busy ? "Working…" : "Refine with preference search"}
          </button>
        </div>
      </div>

      {groups.map((group, groupIndex) => (
        <section key={`${groupIndex}-${group.label}`} className="candidate-group">
          <h3>
            <span className="group-badge">{group.label || "Retrieval direction"}</span>
          </h3>
          <div className="candidate-grid">
            {group.items.map((candidate) => {
              const isSelected = selected.has(candidate.gallery_index);
              return (
                <button
                  key={candidate.gallery_index}
                  type="button"
                  className={`candidate-card plain ${isSelected ? "selected" : ""}`}
                  onClick={() => onToggle(candidate)}
                  disabled={busy}
                  aria-pressed={isSelected}
                >
                  <span className="candidate-slot">#{candidate.slot}</span>
                  <img
                    src={candidate.image_url}
                    alt={`candidate ${candidate.slot}`}
                    loading="lazy"
                  />
                  {isSelected && <span className="candidate-check">✓</span>}
                </button>
              );
            })}
          </div>
        </section>
      ))}
    </div>
  );
}
