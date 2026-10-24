import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  cacheComponents: true,
  partialPrefetching: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
  // FastAPI routes are registered with trailing slashes (e.g. `/health/`);
  // without this Next.js 308-redirects them to the slashless form, which
  // FastAPI would then redirect back, producing a loop through the proxy.
  skipTrailingSlashRedirect: true,
  // Next's gzip middleware buffers proxied responses, which collapses an SSE
  // stream into a single burst at the end. Compression belongs to the reverse
  // proxy in front of the app, not to the rewrite layer.
  compress: false,
  experimental: {
    // Rewrites buffer the request body and silently truncate it at 10MB, which
    // would corrupt multipart uploads. Must exceed the 20MB per-file limit in
    // src/lib/ingest.ts plus multipart framing overhead.
    proxyClientMaxBodySize: "25mb",
  },
  async rewrites() {
    const apiBase = process.env.API_INTERNAL_URL || "http://localhost:8000";
    return [
      {
        source: "/api/v1/:path(.*)",
        destination: `${apiBase}/api/v1/:path`,
      },
    ];
  },
};

export default nextConfig;
