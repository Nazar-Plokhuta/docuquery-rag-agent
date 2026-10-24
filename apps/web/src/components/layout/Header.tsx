"use client";

import { Menu } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

import { HealthStatus } from "@/components/layout/HealthStatus";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { useHealth } from "@/hooks/use-health";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { href: "/chat", label: "Chat" },
  { href: "/ingest", label: "Ingest" },
  { href: "/telemetry", label: "Telemetry" },
] as const;

function EnvironmentBadge() {
  const state = useHealth();
  // Prefer the environment the API reports; fall back to the Next.js build
  // mode so the badge is still meaningful while the API is unreachable.
  const environment =
    state.status === "online"
      ? state.health.environment
      : process.env.NODE_ENV === "production"
        ? "production"
        : "development";

  return (
    <Badge
      variant="outline"
      className={cn(
        "hidden capitalize md:inline-flex",
        environment === "production"
          ? "border-amber-500/30 text-amber-400"
          : "text-muted-foreground",
      )}
    >
      {environment}
    </Badge>
  );
}

function NavLinks({
  orientation,
  onNavigate,
}: {
  orientation: "horizontal" | "vertical";
  onNavigate?: () => void;
}) {
  const pathname = usePathname();

  return (
    <nav
      aria-label="Primary"
      className={cn(
        "flex gap-1",
        orientation === "vertical" ? "flex-col" : "items-center",
      )}
    >
      {NAV_ITEMS.map(({ href, label }) => {
        const active = pathname === href || pathname.startsWith(`${href}/`);
        return (
          <Link
            key={href}
            href={href}
            onClick={onNavigate}
            aria-current={active ? "page" : undefined}
            className={cn(
              "rounded-md px-3 py-1.5 text-sm transition-colors",
              active
                ? "bg-secondary text-foreground"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {label}
          </Link>
        );
      })}
    </nav>
  );
}

export function Header() {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <header className="sticky top-0 z-40 border-b border-border bg-background/80 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-7xl items-center gap-4 px-4 sm:px-6">
        <Link href="/chat" className="flex items-center gap-2 font-semibold">
          <span
            aria-hidden
            className="grid size-6 place-items-center rounded-md bg-foreground font-mono text-xs text-background"
          >
            D
          </span>
          <span className="hidden sm:inline">DocuQuery Studio</span>
        </Link>
        <EnvironmentBadge />

        <div className="ml-4 hidden md:block">
          <NavLinks orientation="horizontal" />
        </div>

        <div className="ml-auto flex items-center gap-2">
          <HealthStatus />
          <Sheet open={menuOpen} onOpenChange={setMenuOpen}>
            <SheetTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="md:hidden"
                aria-label="Open navigation"
              >
                <Menu />
              </Button>
            </SheetTrigger>
            <SheetContent side="left">
              <SheetHeader>
                <SheetTitle>DocuQuery Studio</SheetTitle>
                <SheetDescription className="sr-only">
                  Navigate between Studio sections
                </SheetDescription>
              </SheetHeader>
              <div className="px-4">
                <NavLinks
                  orientation="vertical"
                  onNavigate={() => setMenuOpen(false)}
                />
              </div>
            </SheetContent>
          </Sheet>
        </div>
      </div>
    </header>
  );
}
