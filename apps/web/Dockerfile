# ============================================================
# DocuQuery Studio — Dockerfile
# Target: minimal, non-root Next.js standalone runtime
# Base:   node:22-alpine
# Build context: apps/web (see docker/docker-compose.yml)
# Dependencies: resolved from pnpm-lock.yaml via `pnpm install --frozen-lockfile`
# ============================================================

# --------------- Stage 1: dependency install ---------------
FROM node:22-alpine AS deps

# libc6-compat covers native modules built against glibc (e.g. sharp, SWC).
RUN apk add --no-cache libc6-compat

# Corepack provisions the exact pnpm release pinned in `packageManager`.
RUN corepack enable

WORKDIR /app

# Manifests only, so the install layer is reused until dependencies change.
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile

# --------------- Stage 2: production build ---------------
FROM node:22-alpine AS builder

RUN apk add --no-cache libc6-compat && corepack enable

WORKDIR /app

# Next.js freezes rewrite destinations into the routes manifest at build time,
# so the API origin must be known here rather than at container start.
ARG API_INTERNAL_URL=http://api:8000
ENV API_INTERNAL_URL=${API_INTERNAL_URL} \
    NEXT_TELEMETRY_DISABLED=1

COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN pnpm build

# --------------- Stage 3: lean runtime image ---------------
FROM node:22-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME="0.0.0.0"

# Security: run as an unprivileged user.
RUN addgroup --system --gid 1001 nodejs \
    && adduser --system --uid 1001 --ingroup nodejs nextjs

# The standalone bundle excludes `public/` and `.next/static`; they are
# normally served from a CDN, so they must be copied in explicitly.
COPY --from=builder --chown=nextjs:nodejs /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

# 127.0.0.1, not localhost: Alpine resolves localhost to ::1 first and the
# server listens on IPv4 only (HOSTNAME=0.0.0.0), so BusyBox wget would be refused.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD wget -q -O /dev/null http://127.0.0.1:3000/api/v1/health/ || exit 1

CMD ["node", "server.js"]
