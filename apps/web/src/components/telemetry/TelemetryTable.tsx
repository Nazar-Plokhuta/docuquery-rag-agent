import { Search } from "lucide-react";

import { LatencyBadge } from "@/components/telemetry/LatencyBadge";
import { TokenBar } from "@/components/telemetry/TokenBar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { TelemetryRecord } from "@/lib/api/client";
import { formatTimestampUtc } from "@/lib/telemetry";

interface TelemetryTableProps {
  records: TelemetryRecord[];
  onInspect: (record: TelemetryRecord) => void;
}

function QueryCell({ text }: { text: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="block max-w-xs cursor-default truncate lg:max-w-md">
          {text}
        </span>
      </TooltipTrigger>
      <TooltipContent className="max-w-md whitespace-pre-wrap break-words">
        {text}
      </TooltipContent>
    </Tooltip>
  );
}

export function TelemetryTable({ records, onInspect }: TelemetryTableProps) {
  return (
    <div className="rounded-lg border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Timestamp (UTC)</TableHead>
            <TableHead>Model</TableHead>
            <TableHead>Query</TableHead>
            <TableHead>Latency</TableHead>
            <TableHead>Tokens</TableHead>
            <TableHead className="text-right">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {records.map((record) => (
            <TableRow key={record.request_id}>
              <TableCell className="font-mono text-xs whitespace-nowrap tabular-nums">
                {formatTimestampUtc(record.created_at)}
              </TableCell>
              <TableCell>
                <Badge variant="outline" className="font-mono">
                  {record.model_name}
                </Badge>
              </TableCell>
              <TableCell>
                <QueryCell text={record.query_text} />
              </TableCell>
              <TableCell>
                <LatencyBadge ms={record.latency_ms} />
              </TableCell>
              <TableCell>
                <div className="flex w-28 flex-col gap-1.5">
                  <Badge variant="secondary" className="font-mono tabular-nums">
                    {record.total_tokens.toLocaleString("en-US")}
                  </Badge>
                  <TokenBar
                    promptTokens={record.prompt_tokens}
                    completionTokens={record.completion_tokens}
                  />
                </div>
              </TableCell>
              <TableCell className="text-right">
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => onInspect(record)}
                  aria-label={`Inspect trace ${record.request_id}`}
                >
                  <Search />
                  Inspect
                </Button>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
