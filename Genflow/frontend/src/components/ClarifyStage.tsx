import { useEffect, useState } from "react";
import type { RuntimePlan } from "../types";

interface ClarifyStageProps {
  plan: RuntimePlan;
  busy: boolean;
  onSubmit: (answers: string[]) => void;
  onSkip: () => void;
}

export default function ClarifyStage({
  plan,
  busy,
  onSubmit,
  onSkip,
}: ClarifyStageProps) {
  const questions = plan.clarification_questions;
  const [answers, setAnswers] = useState<string[]>([]);

  useEffect(() => {
    setAnswers(questions.map(() => ""));
  }, [questions.join("|")]);

  const setAnswer = (index: number, value: string) => {
    setAnswers((previous) => {
      const next = [...previous];
      next[index] = value;
      return next;
    });
  };

  return (
    <div className="stage clarify-stage">
      <h1>The agent needs a few details</h1>
      <p className="lede">
        Your answers are appended to the intent, which makes the retrieval more
        accurate. Leaving any field blank — or clicking “I&apos;m not sure” — closes
        clarification and lets the agent decide on its own.
      </p>

      {plan.reasoning_summary && (
        <p className="agent-note">{plan.reasoning_summary}</p>
      )}

      <ol className="question-list">
        {questions.map((question, index) => (
          <li key={`${index}-${question}`}>
            <label htmlFor={`q-${index}`}>{question}</label>
            <input
              id={`q-${index}`}
              type="text"
              value={answers[index] ?? ""}
              placeholder="Type your answer, or leave blank if unsure"
              onChange={(event) => setAnswer(index, event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault();
                  if (answers.some((answer) => answer.trim())) {
                    onSubmit(answers);
                  }
                }
              }}
            />
          </li>
        ))}
      </ol>

      <div className="compose-actions">
        <button
          type="button"
          className="primary"
          disabled={busy || !answers.some((answer) => answer.trim())}
          onClick={() => onSubmit(answers)}
        >
          {busy ? "Submitting…" : "Submit answers"}
        </button>
        <button type="button" className="ghost" disabled={busy} onClick={onSkip}>
          I&apos;m not sure — let the agent decide
        </button>
      </div>
    </div>
  );
}
