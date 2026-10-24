"use client";

import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useHealth } from "@/hooks/use-health";
import { cn } from "@/lib/utils";

function StatusDot({
  className,
  pulse = false,
}: {
  className: string;
  pulse?: boolean;
}) {
  return (
    <span className="relative flex size-2">
      {pulse && (
        <span
          className={cn(
            "absolute inline-flex size-full animate-ping rounded-full opacity-60",
            className,
          )}
        />
      )}
      <span className={cn("relative inline-flex size-2 rounded-full", className)} />
    </span>
  );
}

export function HealthStatus() {
  const state = useHealth();

  if (state.status === "offline") {
    return (
      <Badge
        variant="outline"
        className="gap-2 border-red-500/30 bg-red-500/10 text-red-400"
        role="status"
      >
        <StatusDot className="bg-red-500" />
        Offline / API Disconnected
      </Badge>
    );
  }

  if (state.status === "checking") {
    return (
      <Badge
        variant="outline"
        className="gap-2 text-muted-foreground"
        role="status"
      >
        <StatusDot className="bg-zinc-500" />
        Connecting…
      </Badge>
    );
  }

  const { health } = state;
  return (
    <div className="flex items-center gap-2" role="status">
      <Tooltip>
        <TooltipTrigger asChild>
          <Badge
            variant="outline"
            className="gap-2 border-emerald-500/30 bg-emerald-500/10 text-emerald-400"
          >
            <StatusDot className="bg-emerald-500" pulse />v{health.version}
          </Badge>
        </TooltipTrigger>
        <TooltipContent>
          {health.app_name} · {health.environment}
        </TooltipContent>
      </Tooltip>
      <Tooltip>
        <TooltipTrigger asChild>
          <Badge variant="secondary" className="hidden font-mono sm:inline-flex">
            {health.active_model}
          </Badge>
        </TooltipTrigger>
        <TooltipContent>Embeddings: {health.embedding_model}</TooltipContent>
      </Tooltip>
    </div>
  );
}
