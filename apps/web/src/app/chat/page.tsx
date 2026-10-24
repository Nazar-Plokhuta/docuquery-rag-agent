import type { Metadata } from "next";

import { ChatStudio } from "@/components/chat/ChatStudio";

export const metadata: Metadata = { title: "Chat" };

export default function ChatPage() {
  return <ChatStudio />;
}
