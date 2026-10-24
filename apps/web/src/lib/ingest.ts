// Mirrors `_ALLOWED_EXTENSIONS` in apps/api/src/api/routes/ingestion.py. The
// server stays authoritative (415); this copy only avoids a doomed round trip.
export const ALLOWED_EXTENSIONS = [
  ".md",
  ".txt",
  ".pdf",
  ".docx",
  ".csv",
  ".xlsx",
  ".html",
  ".htm",
] as const;

export const MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024;

// Each ingestion is an embedding-bound request; two in flight keeps the
// provider's rate limit and the API's worker pool comfortable.
export const MAX_CONCURRENT_UPLOADS = 2;

export type IngestStatus =
  | "pending"
  | "uploading"
  | "success"
  | "warning"
  | "error";

export interface IngestItem {
  id: string;
  filename: string;
  size: number;
  status: IngestStatus;
  chunksIngested: number | null;
  errorMessage: string | null;
}

export interface RejectedFile {
  id: string;
  filename: string;
  reason: string;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function getExtension(filename: string): string {
  const dot = filename.lastIndexOf(".");
  return dot === -1 ? "" : filename.slice(dot).toLowerCase();
}

/** Returns a user-facing rejection reason, or `null` when the file is acceptable. */
export function validateFile(file: File): string | null {
  const extension = getExtension(file.name);
  if (!(ALLOWED_EXTENSIONS as readonly string[]).includes(extension)) {
    return extension
      ? `Unsupported file type "${extension}"`
      : "Unsupported file type (no extension)";
  }
  if (file.size > MAX_FILE_SIZE_BYTES) {
    return `Exceeds the ${formatBytes(MAX_FILE_SIZE_BYTES)} limit (${formatBytes(file.size)})`;
  }
  return null;
}
