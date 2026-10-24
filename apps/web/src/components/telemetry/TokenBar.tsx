import { promptSharePercent } from "@/lib/telemetry";
import { cn } from "@/lib/utils";

interface TokenBarProps {
  promptTokens: number;
  completionTokens: number;
  className?: string;
}

/** Segmented bar: prompt tokens (sky) followed by completion tokens (violet). */
export function TokenBar({
  promptTokens,
  completionTokens,
  className,
}: TokenBarProps) {
  const promptShare = promptSharePercent(promptTokens, completionTokens);
  const hasTokens = promptTokens + completionTokens > 0;

  return (
    <div
      role="img"
      aria-label={`${promptTokens} prompt tokens, ${completionTokens} completion tokens`}
      title={`Prompt ${promptTokens.toLocaleString("en-US")} · Completion ${completionTokens.toLocaleString("en-US")}`}
      className={cn(
        "flex h-1.5 w-full overflow-hidden rounded-full bg-muted",
        className,
      )}
    >
      {hasTokens && (
        <>
          <div className="bg-sky-400" style={{ width: `${promptShare}%` }} />
          <div className="flex-1 bg-violet-400" />
        </>
      )}
    </div>
  );
}
