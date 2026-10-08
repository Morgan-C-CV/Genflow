import { useState } from "react";

const EXAMPLES = [
  "Two colossal star destroyers colliding in deep space, explosions and debris, cinematic lighting, hyperrealistic",
  "A serene ghibli-style landscape with pastel skies and lush meadows",
  "Cyberpunk rainy night street, neon reflections, a girl holding an umbrella, cinematic",
];

interface ComposeStageProps {
  busy: boolean;
  onSubmit: (intent: string) => void;
}

export default function ComposeStage({ busy, onSubmit }: ComposeStageProps) {
  const [intent, setIntent] = useState("");

  const submit = () => {
    const trimmed = intent.trim();
    if (!trimmed || busy) return;
    onSubmit(trimmed);
  };

  return (
    <div className="stage compose-stage">
      <h1>Describe what you want to create</h1>
      <p className="lede">
        Genflow plans your intent, asks follow-up questions when needed, retrieves
        candidate references from the gallery, then converts the reference you pick
        into workflow JSON that can be pushed to ComfyUI.
      </p>

      <textarea
        className="intent-input"
        value={intent}
        placeholder="e.g. two colossal star destroyers colliding in deep space, explosions and debris, cinematic lighting…"
        rows={6}
        onChange={(event) => setIntent(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
            event.preventDefault();
            submit();
          }
        }}
      />

      <div className="compose-actions">
        <button type="button" className="primary" onClick={submit} disabled={busy || !intent.trim()}>
          {busy ? "Agent is running…" : "Start Genflow Agent"}
        </button>
        <span className="muted small">⌘/Ctrl + Enter to submit</span>
      </div>

      <div className="examples">
        <span className="axis-label">Try one</span>
        <ul className="chips">
          {EXAMPLES.map((example) => (
            <li key={example}>
              <button type="button" className="chip chip-button" onClick={() => setIntent(example)}>
                {example.slice(0, 42)}…
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
