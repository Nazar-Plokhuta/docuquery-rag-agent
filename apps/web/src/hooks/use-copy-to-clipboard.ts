"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const RESET_DELAY_MS = 1_500;

export type CopyStatus = "idle" | "copied" | "failed";

export function useCopyToClipboard(): {
  status: CopyStatus;
  copy: (text: string) => Promise<void>;
} {
  const [status, setStatus] = useState<CopyStatus>("idle");
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(
    () => () => {
      if (timer.current !== null) clearTimeout(timer.current);
    },
    [],
  );

  const copy = useCallback(async (text: string) => {
    let next: CopyStatus = "copied";
    try {
      // Undefined outside secure contexts (plain-http deployments).
      await navigator.clipboard.writeText(text);
    } catch {
      next = "failed";
    }
    setStatus(next);
    if (timer.current !== null) clearTimeout(timer.current);
    timer.current = setTimeout(() => setStatus("idle"), RESET_DELAY_MS);
  }, []);

  return { status, copy };
}
