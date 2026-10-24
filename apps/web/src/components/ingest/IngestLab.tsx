"use client";

import { RotateCcw, Trash2 } from "lucide-react";

import { IngestItemCard } from "@/components/ingest/IngestItemCard";
import { RejectedFilesAlert } from "@/components/ingest/RejectedFilesAlert";
import { UploadDropzone } from "@/components/ingest/UploadDropzone";
import { Button } from "@/components/ui/button";
import { useIngestQueue } from "@/hooks/use-ingest-queue";

export function IngestLab() {
  const {
    items,
    rejections,
    addFiles,
    retryFailed,
    clearCompleted,
    dismissRejections,
  } = useIngestQueue();

  const failedCount = items.filter((item) => item.status === "error").length;
  const completedCount = items.filter(
    (item) => item.status === "success" || item.status === "warning",
  ).length;
  const activeCount = items.filter(
    (item) => item.status === "pending" || item.status === "uploading",
  ).length;

  return (
    <div className="space-y-4">
      <UploadDropzone onFiles={addFiles} />
      <RejectedFilesAlert
        rejections={rejections}
        onDismiss={dismissRejections}
      />

      {items.length > 0 && (
        <section aria-label="Upload queue" className="space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="text-sm text-muted-foreground" aria-live="polite">
              {items.length} {items.length === 1 ? "file" : "files"}
              {activeCount > 0 && ` · ${activeCount} in progress`}
              {failedCount > 0 && ` · ${failedCount} failed`}
            </p>
            <div className="flex gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                disabled={failedCount === 0}
                onClick={retryFailed}
              >
                <RotateCcw />
                Retry failed
              </Button>
              <Button
                type="button"
                variant="outline"
                size="sm"
                disabled={completedCount === 0}
                onClick={clearCompleted}
              >
                <Trash2 />
                Clear completed
              </Button>
            </div>
          </div>
          <ul className="space-y-2">
            {items.map((item) => (
              <li key={item.id}>
                <IngestItemCard item={item} />
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
