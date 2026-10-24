"use client";

import { createElement, useMemo, type ComponentProps } from "react";
import Markdown, { defaultUrlTransform } from "react-markdown";
import remarkGfm from "remark-gfm";

import { CitationBadge } from "@/components/chat/CitationBadge";
import {
  citationKey,
  findCitationMarkers,
  type ChatCitation,
} from "@/lib/chat";
import { cn } from "@/lib/utils";

interface MdNode {
  type: string;
  value?: string;
  url?: string;
  children?: MdNode[];
}

const CITATION_SCHEME = "cite:";

function citationHref({ source, section }: ChatCitation): string {
  return CITATION_SCHEME + new URLSearchParams({ source, section }).toString();
}

function parseCitationHref(href: string): ChatCitation {
  const params = new URLSearchParams(href.slice(CITATION_SCHEME.length));
  return {
    source: params.get("source") ?? "",
    section: params.get("section") ?? "",
    chunkId: null,
  };
}

function splitTextNode(value: string): MdNode[] {
  const markers = findCitationMarkers(value);
  if (markers.length === 0) return [{ type: "text", value }];

  const nodes: MdNode[] = [];
  let cursor = 0;
  for (const { start, end, citation } of markers) {
    if (start > cursor) {
      nodes.push({ type: "text", value: value.slice(cursor, start) });
    }
    nodes.push({
      type: "link",
      url: citationHref(citation),
      children: [{ type: "text", value: citation.source }],
    });
    cursor = end;
  }
  if (cursor < value.length) {
    nodes.push({ type: "text", value: value.slice(cursor) });
  }
  return nodes;
}

function rewriteCitations(node: MdNode): void {
  // Links are left alone: a badge (a link) must never nest inside another.
  if (!node.children || node.type === "link") return;
  node.children = node.children.flatMap((child) => {
    if (child.type === "text" && child.value) return splitTextNode(child.value);
    rewriteCitations(child);
    return [child];
  });
}

/**
 * Turns `[Source: x, Section: y]` markers in text nodes into link nodes with a
 * private `cite:` scheme. Working on the parsed tree (not the raw string)
 * keeps markers inside code spans and fences verbatim.
 */
function remarkCitations() {
  return (tree: MdNode) => rewriteCitations(tree);
}

const REMARK_PLUGINS = [remarkGfm, remarkCitations];

// Only the app's own scheme is added to react-markdown's default allow-list
// (http, https, mailto, ...); `javascript:` and friends stay stripped.
function urlTransform(url: string): string {
  return url.startsWith(CITATION_SCHEME) ? url : defaultUrlTransform(url);
}

type Components = NonNullable<ComponentProps<typeof Markdown>["components"]>;

interface StyledProps {
  node?: unknown;
  className?: string;
}

// react-markdown passes the hast `node` to every component; spreading it onto
// a DOM element would leak an object-valued attribute.
function styled(tag: string, baseClassName: string) {
  return function Styled({ node, className, ...props }: StyledProps) {
    void node;
    return createElement(tag, {
      ...props,
      className: cn(baseClassName, className),
    });
  };
}

const STATIC_COMPONENTS: Components = {
  p: styled("p", "leading-relaxed [&:not(:first-child)]:mt-3"),
  h1: styled("h3", "mt-4 text-base font-semibold"),
  h2: styled("h3", "mt-4 text-base font-semibold"),
  h3: styled("h4", "mt-3 text-sm font-semibold"),
  ul: styled("ul", "mt-3 list-disc space-y-1 pl-5"),
  ol: styled("ol", "mt-3 list-decimal space-y-1 pl-5"),
  blockquote: styled(
    "blockquote",
    "mt-3 border-l-2 border-border pl-3 text-muted-foreground",
  ),
  pre: styled(
    "pre",
    "mt-3 overflow-x-auto rounded-md border bg-muted/50 p-3 font-mono text-xs",
  ),
  code: styled(
    "code",
    "rounded bg-muted/60 px-1 py-0.5 font-mono text-[0.85em] [pre_&]:bg-transparent [pre_&]:p-0",
  ),
  table: ({ node, ...props }: StyledProps) => {
    void node;
    return (
      <div className="mt-3 overflow-x-auto">
        {createElement("table", {
          ...props,
          className: "w-full border-collapse text-xs",
        })}
      </div>
    );
  },
  th: styled("th", "border border-border px-2 py-1 text-left font-medium"),
  td: styled("td", "border border-border px-2 py-1"),
  // Model output is untrusted (documents can carry prompt injections): a
  // remote image would let it exfiltrate data through the request URL.
  img: ({ alt }) => (alt ? <span>{alt}</span> : null),
};

interface AnswerMarkdownProps {
  content: string;
  /** Authoritative citations (inspect mode) used to enrich inline markers. */
  citations: ChatCitation[] | null;
  onCitationSelect: (citation: ChatCitation) => void;
}

export function AnswerMarkdown({
  content,
  citations,
  onCitationSelect,
}: AnswerMarkdownProps) {
  const components = useMemo<Components>(() => {
    const known = new Map(
      (citations ?? []).map((citation) => [citationKey(citation), citation]),
    );
    return {
      ...STATIC_COMPONENTS,
      a: ({ node, href, children, ...props }) => {
        void node;
        if (href?.startsWith(CITATION_SCHEME)) {
          const parsed = parseCitationHref(href);
          return (
            <CitationBadge
              citation={known.get(citationKey(parsed)) ?? parsed}
              onSelect={onCitationSelect}
            />
          );
        }
        return (
          <a
            href={href}
            target="_blank"
            rel="noopener noreferrer"
            className="text-sky-300 underline underline-offset-2"
            {...props}
          >
            {children}
          </a>
        );
      },
    };
  }, [citations, onCitationSelect]);

  return (
    <Markdown
      remarkPlugins={REMARK_PLUGINS}
      urlTransform={urlTransform}
      components={components}
    >
      {content}
    </Markdown>
  );
}
