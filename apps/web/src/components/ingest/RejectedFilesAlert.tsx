"use client";

import { TriangleAlert, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import type { RejectedFile } from "@/lib/ingest";

interface RejectedFilesAlertProps {
  rejections: RejectedFile[];
  onDismiss: () => void;
}

export function RejectedFilesAlert({
  rejections,
  onDismiss,
}: RejectedFilesAlertProps) {
  if (rejections.length === 0) return null;

  return (
    <div
      role="alert"
      className="flex items-start gap-3 rounded-lg border border-rose-500/30 bg-rose-500/10 p-3 text-sm"
    >
      <TriangleAlert
        aria-hidden
        className="mt-0.5 size-4 shrink-0 text-rose-400"
      />
      <div className="min-w-0 flex-1 space-y-1">
        <p className="font-medium text-rose-300">
          {rejections.length === 1
            ? "1 file was not added"
            : `${rejections.length} files were not added`}
        </p>
        <ul className="space-y-0.5 text-xs text-muted-foreground">
          {rejections.map((rejection) => (
            <li key={rejection.id} className="truncate">
              <span className="font-mono text-foreground">
                {rejection.filename}
              </span>{" "}
              — {rejection.reason}
            </li>
          ))}
        </ul>
      </div>
      <Button
        type="button"
        variant="ghost"
        size="icon-xs"
        aria-label="Dismiss"
        onClick={onDismiss}
      >
        <X />
      </Button>
    </div>
  );
}
