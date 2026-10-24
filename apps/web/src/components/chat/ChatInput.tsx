"use client";

import { ArrowUp, Square } from "lucide-react";
import {
  useLayoutEffect,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";

import { Button } from "@/components/ui/button";

const MAX_HEIGHT_PX = 200;

interface ChatInputProps {
  isGenerating: boolean;
  onSubmit: (text: string) => void;
  onStop: () => void;
}

export function ChatInput({ isGenerating, onSubmit, onStop }: ChatInputProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useLayoutEffect(() => {
    const element = textareaRef.current;
    if (!element) return;
    // Collapse first so the box can shrink after text is deleted or sent.
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, MAX_HEIGHT_PX)}px`;
  }, [value]);

  const canSend = value.trim() !== "" && !isGenerating;

  const submit = () => {
    if (!canSend) return;
    onSubmit(value);
    setValue("");
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    // `isComposing`: Enter confirms an IME candidate and must not send.
    if (
      event.key !== "Enter" ||
      event.shiftKey ||
      event.nativeEvent.isComposing
    ) {
      return;
    }
    event.preventDefault();
    submit();
  };

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        submit();
      }}
      className="flex items-end gap-2 rounded-xl border bg-card p-2 focus-within:ring-[3px] focus-within:ring-ring/40"
    >
      <textarea
        ref={textareaRef}
        value={value}
        rows={1}
        autoFocus
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Ask a question about your documents…"
        aria-label="Question"
        className="max-h-50 min-h-9 flex-1 resize-none bg-transparent px-2 py-1.5 text-sm outline-none placeholder:text-muted-foreground"
      />
      {isGenerating ? (
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={onStop}
          aria-label="Stop generating"
        >
          <Square className="fill-current" />
          Stop
        </Button>
      ) : (
        <Button
          type="submit"
          size="icon"
          disabled={!canSend}
          aria-label="Send message"
        >
          <ArrowUp />
        </Button>
      )}
    </form>
  );
}
