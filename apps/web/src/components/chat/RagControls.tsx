"use client";

import { RotateCcw } from "lucide-react";
import { useId } from "react";

import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  DEFAULT_RAG_PARAMS,
  SCORE_THRESHOLD_RANGE,
  TOP_K_RANGE,
  type ChatMode,
  type RagParams,
} from "@/lib/chat";

interface RagControlsProps {
  mode: ChatMode;
  onModeChange: (mode: ChatMode) => void;
  params: RagParams;
  onParamsChange: (params: RagParams) => void;
  disabled: boolean;
}

function isChatMode(value: string): value is ChatMode {
  return value === "stream" || value === "inspect";
}

export function RagControls({
  mode,
  onModeChange,
  params,
  onParamsChange,
  disabled,
}: RagControlsProps) {
  // The controls render in both the desktop aside and the mobile sheet.
  const idPrefix = useId();
  const topKId = `${idPrefix}-top-k`;
  const thresholdId = `${idPrefix}-score-threshold`;
  const isDefault =
    params.topK === DEFAULT_RAG_PARAMS.topK &&
    params.scoreThreshold === DEFAULT_RAG_PARAMS.scoreThreshold;

  return (
    <div className="space-y-5">
      <div className="space-y-2">
        <h2 className="text-sm font-medium">Response mode</h2>
        <Tabs
          value={mode}
          onValueChange={(value) => {
            if (isChatMode(value)) onModeChange(value);
          }}
        >
          <TabsList className="w-full">
            <TabsTrigger value="stream" disabled={disabled}>
              Stream
            </TabsTrigger>
            <TabsTrigger value="inspect" disabled={disabled}>
              Inspect
            </TabsTrigger>
          </TabsList>
        </Tabs>
        <p className="text-xs text-muted-foreground">
          {mode === "stream"
            ? "Tokens appear as they are generated."
            : "Waits for the full answer and reports citations, latency and token usage."}
        </p>
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label htmlFor={topKId} className="text-sm font-medium">
            top_k
          </label>
          <span className="font-mono text-xs tabular-nums">{params.topK}</span>
        </div>
        <Slider
          id={topKId}
          aria-label="top_k"
          min={TOP_K_RANGE.min}
          max={TOP_K_RANGE.max}
          step={1}
          value={[params.topK]}
          disabled={disabled}
          onValueChange={([topK]) => onParamsChange({ ...params, topK })}
        />
        <p className="text-xs text-muted-foreground">
          Candidate chunks fetched from the vector store.
        </p>
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label htmlFor={thresholdId} className="text-sm font-medium">
            score_threshold
          </label>
          <span className="font-mono text-xs tabular-nums">
            {params.scoreThreshold.toFixed(2)}
          </span>
        </div>
        <Slider
          id={thresholdId}
          aria-label="score_threshold"
          min={SCORE_THRESHOLD_RANGE.min}
          max={SCORE_THRESHOLD_RANGE.max}
          step={SCORE_THRESHOLD_RANGE.step}
          value={[params.scoreThreshold]}
          disabled={disabled}
          onValueChange={([scoreThreshold]) =>
            onParamsChange({ ...params, scoreThreshold })
          }
        />
        <p className="text-xs text-muted-foreground">
          Minimum similarity for a chunk to reach the prompt. Higher values
          refuse more often.
        </p>
      </div>

      <Button
        type="button"
        variant="ghost"
        size="sm"
        disabled={disabled || isDefault}
        onClick={() => onParamsChange(DEFAULT_RAG_PARAMS)}
      >
        <RotateCcw />
        Reset to defaults
      </Button>
    </div>
  );
}
