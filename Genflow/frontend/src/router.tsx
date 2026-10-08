/**
 * Minimal history-API router.
 *
 * The app only needs two entry points (the studio and the refine showcase), so
 * this avoids pulling in a routing dependency. Vite's dev server already falls
 * back to index.html for unknown paths, which makes /showcase/refine work on a
 * hard refresh.
 */

import { useEffect, useState } from "react";

export function normalizePath(path: string): string {
  const trimmed = path.replace(/\/+$/, "");
  return trimmed === "" ? "/" : trimmed;
}

export function navigate(to: string): void {
  if (normalizePath(window.location.pathname) === normalizePath(to)) return;
  window.history.pushState({}, "", to);
  window.dispatchEvent(new PopStateEvent("popstate"));
}

export function usePathname(): string {
  const [path, setPath] = useState(() => normalizePath(window.location.pathname));

  useEffect(() => {
    const onPop = () => setPath(normalizePath(window.location.pathname));
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  return path;
}

interface LinkProps {
  to: string;
  className?: string;
  children: React.ReactNode;
}

export function Link({ to, className, children }: LinkProps) {
  return (
    <a
      href={to}
      className={className}
      onClick={(event) => {
        if (event.metaKey || event.ctrlKey || event.shiftKey) return;
        event.preventDefault();
        navigate(to);
      }}
    >
      {children}
    </a>
  );
}
