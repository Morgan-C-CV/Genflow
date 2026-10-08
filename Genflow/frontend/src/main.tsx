import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { usePathname } from "./router";
import RefineShowcase from "./showcase/RefineShowcase";
import "./styles.css";

/** Two entry points: the studio, and the refine showcase. */
function Root() {
  const path = usePathname();
  if (path === "/showcase/refine") return <RefineShowcase />;
  return <App />;
}

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>
);
