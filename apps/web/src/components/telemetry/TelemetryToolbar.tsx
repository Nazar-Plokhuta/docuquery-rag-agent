"use client";

import { RefreshCw } from "lucide-react";
import { useId } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import {
  AUTO_REFRESH_INTERVAL_MS,
  isLimitOption,
  LIMIT_OPTIONS,
} from "@/lib/telemetry";

interface TelemetryToolbarProps {
  limit: number;
  onLimitChange: (limit: number) => void;
  recordCount: number;
  isRefreshing: boolean;
  onRefresh: () => void;
  autoRefresh: boolean;
  onAutoRefreshChange: (enabled: boolean) => void;
}

export function TelemetryToolbar({
  limit,
  onLimitChange,
  recordCount,
  isRefreshing,
  onRefresh,
  autoRefresh,
  onAutoRefreshChange,
}: TelemetryToolbarProps) {
  const idPrefix = useId();
  const limitId = `${idPrefix}-limit`;
  const autoId = `${idPrefix}-auto`;

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-3">
      <div className="flex items-center gap-2">
        <label htmlFor={limitId} className="text-sm text-muted-foreground">
          Limit
        </label>
        <Select
          value={String(limit)}
          onValueChange={(value) => {
            const next = Number(value);
            if (isLimitOption(next)) onLimitChange(next);
          }}
        >
          <SelectTrigger id={limitId} size="sm" className="w-24">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {LIMIT_OPTIONS.map((option) => (
              <SelectItem key={option} value={String(option)}>
                {option}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <Button
        type="button"
        variant="outline"
        size="sm"
        onClick={onRefresh}
        disabled={isRefreshing}
      >
        <RefreshCw className={isRefreshing ? "animate-spin" : undefined} />
        Refresh
      </Button>

      <div className="flex items-center gap-2">
        <Switch
          id={autoId}
          checked={autoRefresh}
          onCheckedChange={onAutoRefreshChange}
        />
        <label htmlFor={autoId} className="text-sm">
          Auto-refresh{" "}
          <span className="text-muted-foreground">
            ({AUTO_REFRESH_INTERVAL_MS / 1000}s)
          </span>
        </label>
      </div>

      <Badge variant="secondary" className="ml-auto tabular-nums" aria-live="polite">
        Showing {recordCount} {recordCount === 1 ? "record" : "records"}
      </Badge>
    </div>
  );
}
