"use client";

import { Check, Copy } from "lucide-react";
import type { ComponentProps } from "react";

import { Button } from "@/components/ui/button";
import { useCopyToClipboard } from "@/hooks/use-copy-to-clipboard";

interface CopyButtonProps {
  text: string;
  label: string;
  size?: ComponentProps<typeof Button>["size"];
  /** Hides the visible label; `label` still names the button for screen readers. */
  iconOnly?: boolean;
}

export function CopyButton({
  text,
  label,
  size = "sm",
  iconOnly = false,
}: CopyButtonProps) {
  const { status, copy } = useCopyToClipboard();
  const caption =
    status === "copied" ? "Copied" : status === "failed" ? "Copy failed" : label;

  return (
    <Button
      type="button"
      variant="outline"
      size={size}
      aria-label={label}
      onClick={() => void copy(text)}
    >
      {status === "copied" ? <Check className="text-emerald-400" /> : <Copy />}
      {iconOnly ? <span className="sr-only">{caption}</span> : caption}
    </Button>
  );
}
