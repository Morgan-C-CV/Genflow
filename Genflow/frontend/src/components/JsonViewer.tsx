import { useMemo, useState } from "react";

interface JsonViewerProps {
  value: unknown;
  filename: string;
  label?: string;
  collapsedHeight?: number;
}

export default function JsonViewer({
  value,
  filename,
  label,
  collapsedHeight = 260,
}: JsonViewerProps) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  const text = useMemo(() => JSON.stringify(value, null, 2), [value]);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  };

  const download = () => {
    const blob = new Blob([text], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="json-viewer">
      <div className="json-toolbar">
        <span className="json-label">{label ?? filename}</span>
        <span className="json-actions">
          <span className="muted small">{(text.length / 1024).toFixed(1)} KB</span>
          <button type="button" className="ghost small" onClick={copy}>
            {copied ? "Copied" : "Copy"}
          </button>
          <button type="button" className="ghost small" onClick={download}>
            Download
          </button>
          <button
            type="button"
            className="ghost small"
            onClick={() => setExpanded((previous) => !previous)}
          >
            {expanded ? "Collapse" : "Expand"}
          </button>
        </span>
      </div>
      <pre
        className="json-body"
        style={{ maxHeight: expanded ? "none" : collapsedHeight }}
      >
        {text}
      </pre>
    </div>
  );
}
