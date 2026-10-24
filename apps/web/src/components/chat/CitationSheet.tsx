import { FileText } from "lucide-react";
import type { ReactNode } from "react";

import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import type { ChatCitation } from "@/lib/chat";

interface CitationSheetProps {
  citation: ChatCitation | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

// Chunk ids are `{document_id}#c{index:04d}` (see the API's `Citation` model).
function parseChunkId(chunkId: string): { documentId: string; index: number } | null {
  const match = /^(.+)#c(\d+)$/.exec(chunkId);
  return match ? { documentId: match[1], index: Number(match[2]) } : null;
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="space-y-1">
      <dt className="text-xs font-medium tracking-wide text-muted-foreground uppercase">
        {label}
      </dt>
      <dd className="text-sm break-words">{children}</dd>
    </div>
  );
}

export function CitationSheet({
  citation,
  open,
  onOpenChange,
}: CitationSheetProps) {
  const chunk = citation?.chunkId ? parseChunkId(citation.chunkId) : null;

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right">
        <SheetHeader>
          <SheetTitle className="flex items-center gap-2">
            <FileText aria-hidden className="size-4 text-sky-300" />
            Citation source
          </SheetTitle>
          <SheetDescription>
            The indexed segment this part of the answer was drawn from.
          </SheetDescription>
        </SheetHeader>
        {citation && (
          <dl className="space-y-5 px-4">
            <Field label="Document">
              <span className="font-mono">{citation.source}</span>
            </Field>
            <Field label="Section">
              {citation.section ? (
                citation.section
              ) : (
                <span className="text-muted-foreground">
                  No section heading
                </span>
              )}
            </Field>
            <Field label="Chunk">
              {citation.chunkId ? (
                <div className="space-y-1">
                  <span className="font-mono text-xs">{citation.chunkId}</span>
                  {chunk && (
                    <p className="text-xs text-muted-foreground">
                      Segment {chunk.index} of document{" "}
                      <span className="font-mono">{chunk.documentId}</span>
                    </p>
                  )}
                </div>
              ) : (
                <span className="text-muted-foreground">
                  The streaming endpoint does not report chunk ids. Run the
                  question in Inspect mode to see the exact chunk.
                </span>
              )}
            </Field>
          </dl>
        )}
      </SheetContent>
    </Sheet>
  );
}
