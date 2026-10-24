import type { Metadata } from "next";

import { IngestLab } from "@/components/ingest/IngestLab";

export const metadata: Metadata = { title: "Ingest" };

export default function IngestPage() {
  return (
    <div className="mx-auto w-full max-w-3xl space-y-6 px-4 py-8 sm:px-6">
      <header className="space-y-1">
        <h1 className="text-xl font-semibold tracking-tight">
          Document Ingestion Lab
        </h1>
        <p className="text-sm text-muted-foreground">
          Upload documents to be parsed, chunked, embedded, and indexed into the
          vector store.
        </p>
      </header>
      <IngestLab />
    </div>
  );
}
