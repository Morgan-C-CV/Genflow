import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { usePathname } from "./router";
import ModifyShowcase from "./showcase/ModifyShowcase";
import "./styles.css";

/** Two entry points: the studio, and the refine showcase. */
function Root() {
  const path = usePathname();
  if (path === "/showcase/refine" || path === "/showcase/modify") {
    return <ModifyShowcase />;
  }
  return <App />;
}

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>
);
