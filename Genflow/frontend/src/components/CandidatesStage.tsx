import type { RuntimeCandidate, RuntimeWall } from "../types";

interface CandidatesStageProps {
  wall: RuntimeWall;
  busy: boolean;
  selectedIndex: number | null;
  onSelect: (candidate: RuntimeCandidate) => void;
  onRefresh: () => void;
  onConfirm: () => void;
}

/**
 * Creation stage candidate wall (thesis 4.2.4): 8 expansions x 2 records, with a
 * refresh that blocks everything already shown. Selecting one record is the
 * preference evidence the rest of the pipeline inherits from.
 */
export default function CandidatesStage({
  wall,
  busy,
  selectedIndex,
  onSelect,
  onRefresh,
  onConfirm,
}: CandidatesStageProps) {
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
          <h1>Pick a reference image</h1>
          <p className="lede">
            {wall.candidates.length} images across {groups.length} retrieval
            directions. Your choice becomes the reference the workflow is composed
            from.
          </p>
        </div>
        <button type="button" className="ghost" onClick={onRefresh} disabled={busy}>
          {busy ? "Generating…" : "Shuffle images"}
        </button>
      </div>

      <div className="selection-bar">
        <span className="selection-count">
          {selectedIndex === null
            ? "No image selected"
            : `Selected image #${selectedIndex}`}
        </span>
        <div className="actions">
          <button
            type="button"
            className="primary"
            onClick={onConfirm}
            disabled={busy || selectedIndex === null}
            title={selectedIndex === null ? "Select an image first" : undefined}
          >
            {busy ? "Working…" : "Use this image →"}
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
              const isSelected = selectedIndex === candidate.gallery_index;
              return (
                <button
                  key={candidate.gallery_index}
                  type="button"
                  className={`candidate-card plain ${isSelected ? "selected" : ""}`}
                  onClick={() => onSelect(candidate)}
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
