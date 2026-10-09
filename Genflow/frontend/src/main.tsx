import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./styles.css";

/**
 * `/showcase/refine` renders the same app as `/`. Refine needs an existing result
 * to act on, so that entry point seeds one from a gallery record and opens the
 * modify stage directly; nothing else differs.
 */
const path = window.location.pathname.replace(/\/+$/, "");
const seedRefineFromGallery = path === "/showcase/refine";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App seedRefineFromGallery={seedRefineFromGallery} />
  </React.StrictMode>
);
