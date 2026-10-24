import { FileText } from "lucide-react";

import { cn } from "@/lib/utils";
import type { ChatCitation } from "@/lib/chat";

interface CitationBadgeProps {
  citation: ChatCitation;
  onSelect: (citation: ChatCitation) => void;
  className?: string;
}

export function CitationBadge({
  citation,
  onSelect,
  className,
}: CitationBadgeProps) {
  const { source, section } = citation;
  const label = section ? `${source}, section ${section}` : source;

  return (
    <button
      type="button"
      onClick={() => onSelect(citation)}
      title={label}
      aria-label={`Open citation: ${label}`}
      className={cn(
        "mx-0.5 inline-flex max-w-52 items-center gap-1 rounded-full border border-sky-500/30 bg-sky-500/10 px-2 py-0.5 align-baseline text-xs font-medium text-sky-300 transition-colors hover:bg-sky-500/20 focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none",
        className,
      )}
    >
      <FileText aria-hidden className="size-3 shrink-0" />
      <span className="truncate">{source}</span>
    </button>
  );
}
