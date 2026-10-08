export type Stage = "compose" | "clarify" | "candidates" | "modify" | "workflow";

const STEPS: { key: Stage; label: string }[] = [
  { key: "compose", label: "Intent" },
  { key: "clarify", label: "Clarify" },
  { key: "candidates", label: "Candidates" },
  { key: "modify", label: "Refine" },
  { key: "workflow", label: "Workflow" },
];

interface StepperProps {
  stage: Stage;
  completed: Stage[];
  onJump: (stage: Stage) => void;
}

export default function Stepper({ stage, completed, onJump }: StepperProps) {
  return (
    <nav className="stepper">
      {STEPS.map((step, index) => {
        const isCurrent = step.key === stage;
        const isDone = completed.includes(step.key) && !isCurrent;
        return (
          <button
            key={step.key}
            type="button"
            className={`step ${isCurrent ? "current" : ""} ${isDone ? "done" : ""}`}
            onClick={() => onJump(step.key)}
            disabled={!isCurrent && !isDone && !completed.includes(step.key)}
          >
            <span className="step-index">{isDone ? "✓" : index + 1}</span>
            <span className="step-label">{step.label}</span>
          </button>
        );
      })}
    </nav>
  );
}
