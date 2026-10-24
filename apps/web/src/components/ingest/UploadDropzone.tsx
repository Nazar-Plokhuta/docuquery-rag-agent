"use client";

import { Info, UploadCloud } from "lucide-react";
import { useRef, useState, type DragEvent, type KeyboardEvent } from "react";

import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  ALLOWED_EXTENSIONS,
  MAX_FILE_SIZE_BYTES,
  formatBytes,
} from "@/lib/ingest";
import { cn } from "@/lib/utils";

interface UploadDropzoneProps {
  onFiles: (files: File[]) => void;
}

export function UploadDropzone({ onFiles }: UploadDropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const openPicker = () => inputRef.current?.click();

  const handleDragLeave = (event: DragEvent<HTMLDivElement>) => {
    // Moving over a child element fires dragleave on the zone; ignore it.
    if (event.currentTarget.contains(event.relatedTarget as Node | null)) return;
    setIsDragging(false);
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    onFiles(Array.from(event.dataTransfer.files));
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.target !== event.currentTarget) return;
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openPicker();
    }
  };

  return (
    <div
      role="button"
      tabIndex={0}
      aria-label="Upload documents: drop files here or press Enter to browse"
      onClick={openPicker}
      onKeyDown={handleKeyDown}
      onDragOver={(event) => {
        event.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={cn(
        "flex cursor-pointer flex-col items-center gap-3 rounded-xl border-2 border-dashed px-6 py-12 text-center outline-none transition-colors",
        "focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50",
        isDragging
          ? "border-primary bg-primary/5"
          : "border-border hover:border-muted-foreground/60 hover:bg-accent/30",
      )}
    >
      <UploadCloud
        aria-hidden
        className={cn(
          "size-9",
          isDragging ? "text-primary" : "text-muted-foreground",
        )}
      />
      <div className="space-y-1">
        <p className="text-sm font-medium">
          {isDragging ? "Release to add files" : "Drag & drop documents here"}
        </p>
        <p className="text-xs text-muted-foreground">
          or browse your computer · up to {formatBytes(MAX_FILE_SIZE_BYTES)} each
        </p>
      </div>
      <div className="flex items-center gap-2">
        <Button type="button" variant="outline" size="sm" tabIndex={-1}>
          Browse files
        </Button>
        <Tooltip>
          <TooltipTrigger asChild>
            <span
              className="text-muted-foreground"
              onClick={(event) => event.stopPropagation()}
            >
              <Info aria-label="Supported file types" className="size-4" />
            </span>
          </TooltipTrigger>
          <TooltipContent>
            Supported: {ALLOWED_EXTENSIONS.join(" ")}
          </TooltipContent>
        </Tooltip>
      </div>
      <input
        ref={inputRef}
        type="file"
        multiple
        hidden
        accept={ALLOWED_EXTENSIONS.join(",")}
        onClick={(event) => event.stopPropagation()}
        onChange={(event) => {
          onFiles(Array.from(event.target.files ?? []));
          // Reset so re-selecting the same file still fires `change`.
          event.target.value = "";
        }}
      />
    </div>
  );
}
