import {
  CircleAlert,
  CircleCheck,
  FileText,
  Loader2,
  TriangleAlert,
} from "lucide-react";
import type { ReactNode } from "react";

import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { formatBytes, type IngestItem } from "@/lib/ingest";
import { cn } from "@/lib/utils";

const BADGE_BASE = "border";

function StatusBadge({ item }: { item: IngestItem }): ReactNode {
  switch (item.status) {
    case "pending":
      return <Badge variant="outline">Queued</Badge>;
    case "uploading":
      return (
        <Badge variant="outline" className="border-sky-500/40 text-sky-300">
          <Loader2 className="animate-spin" />
          Ingesting…
        </Badge>
      );
    case "success":
      return (
        <Badge
          className={cn(
            BADGE_BASE,
            "border-emerald-500/30 bg-emerald-500/15 text-emerald-300",
          )}
        >
          <CircleCheck />
          {item.chunksIngested} {item.chunksIngested === 1 ? "chunk" : "chunks"}
        </Badge>
      );
    case "warning":
      return (
        <Tooltip>
          <TooltipTrigger asChild>
            <Badge
              tabIndex={0}
              className={cn(
                BADGE_BASE,
                "border-amber-500/30 bg-amber-500/15 text-amber-300",
              )}
            >
              <TriangleAlert />
              Empty document (0 chunks indexed)
            </Badge>
          </TooltipTrigger>
          <TooltipContent>
            The file was accepted but contained no extractable text, so nothing
            will be retrievable from it.
          </TooltipContent>
        </Tooltip>
      );
    case "error":
      return (
        <Badge
          className={cn(
            BADGE_BASE,
            "border-rose-500/30 bg-rose-500/15 text-rose-300",
          )}
        >
          <CircleAlert />
          Failed
        </Badge>
      );
  }
}

const ACCENT: Record<IngestItem["status"], string> = {
  pending: "border-l-border",
  uploading: "border-l-sky-500",
  success: "border-l-emerald-500",
  warning: "border-l-amber-500",
  error: "border-l-rose-500",
};

export function IngestItemCard({ item }: { item: IngestItem }) {
  return (
    <Card
      className={cn("gap-2 border-l-4 px-4 py-3", ACCENT[item.status])}
      aria-busy={item.status === "uploading"}
    >
      <div className="flex items-center gap-3">
        <FileText aria-hidden className="size-4 shrink-0 text-muted-foreground" />
        <div className="min-w-0 flex-1">
          <p className="truncate font-mono text-sm" title={item.filename}>
            {item.filename}
          </p>
          <p className="text-xs text-muted-foreground">
            {formatBytes(item.size)}
          </p>
        </div>
        <StatusBadge item={item} />
      </div>

      {item.status === "uploading" && (
        <div
          role="progressbar"
          aria-label={`Ingesting ${item.filename}`}
          className="relative h-1 overflow-hidden rounded-full bg-muted"
        >
          {/* Size is unknown until the server responds, so progress is indeterminate. */}
          <div className="absolute inset-y-0 w-1/3 animate-[ingest-slide_1.2s_ease-in-out_infinite] motion-reduce:animate-none rounded-full bg-sky-500" />
        </div>
      )}

      {item.status === "error" && item.errorMessage && (
        <Tooltip>
          <TooltipTrigger asChild>
            <p
              tabIndex={0}
              className="line-clamp-2 text-xs text-rose-300 outline-none"
            >
              {item.errorMessage}
            </p>
          </TooltipTrigger>
          <TooltipContent className="max-w-sm">
            {item.errorMessage}
          </TooltipContent>
        </Tooltip>
      )}
    </Card>
  );
}
