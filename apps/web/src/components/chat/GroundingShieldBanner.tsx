import { ShieldCheck } from "lucide-react";

import type { GuardrailVerdict } from "@/lib/chat";

type Verdict = NonNullable<GuardrailVerdict>;

function describe(verdict: Verdict): { title: string; body: string } {
  if (verdict.kind === "uncited") {
    return {
      title: "No citations attached",
      body: "This answer carries no source citations, so it cannot be traced back to your documents. Treat it as unverified.",
    };
  }
  return verdict.shortCircuited
    ? {
        title: "Grounding guard engaged — zero tokens spent",
        body: "No indexed chunk cleared the similarity threshold, so the model was never called and nothing could be invented. Try lowering the score threshold, raising top_k, or ingesting more documents.",
      }
    : {
        title: "Grounding guard engaged",
        body: "The retrieved context did not contain enough information, so the model declined to answer instead of guessing. Try lowering the score threshold or raising top_k.",
      };
}

export function GroundingShieldBanner({ verdict }: { verdict: Verdict }) {
  const { title, body } = describe(verdict);

  return (
    <div
      role="status"
      className="flex items-start gap-3 rounded-lg border border-amber-500/40 bg-amber-500/10 p-3 text-sm"
    >
      <ShieldCheck
        aria-hidden
        className="mt-0.5 size-5 shrink-0 text-amber-400"
      />
      <div className="min-w-0 space-y-1">
        <p className="font-medium text-amber-300">{title}</p>
        <p className="text-xs leading-relaxed text-amber-100/80">{body}</p>
      </div>
    </div>
  );
}
