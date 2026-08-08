-- Nullain Agent SDK — Supabase/Postgres schema (docs/FUSION_PLAN.md §4, Fase 4)
--
-- `events` is written by nullain.events.postgres_store.PostgresEventStore
-- (the SDK owns that table's shape — see postgres_store.py's module
-- docstring). Every other table here is relational-side application data
-- the *app* (nullain-agent) owns and writes to directly — the SDK does
-- not read or write sessions/users/metadata/traces itself. Run this
-- against a fresh Supabase project (or any Postgres instance) before
-- pointing PostgresEventStore/the app at it.
--
-- Free Tier notes (docs/FUSION_PLAN.md): Supabase's free tier is a single
-- Postgres instance with generous row limits for this table shape —
-- no special scaling considerations needed until launch.

-- ── events ──────────────────────────────────────────────────────────
-- Append-only conversation trajectory, one row per BaseEvent. Mirrors
-- SQLiteEventStore's schema exactly (same columns, same semantics) so a
-- session's replay looks identical regardless of which EventStorePort
-- adapter wrote it. `seq` (not `timestamp`) is the authoritative
-- ordering key — see postgres_store.py / store.py for why.
--
-- PostgresEventStore.initialize() creates this table itself (idempotent,
-- CREATE TABLE IF NOT EXISTS) if it does not already exist — this file is
-- the reference/documentation copy, not a required manual migration step.
CREATE TABLE IF NOT EXISTS events (
    seq         BIGSERIAL PRIMARY KEY,
    id          TEXT NOT NULL UNIQUE,
    session_id  TEXT NOT NULL,
    timestamp   DOUBLE PRECISION NOT NULL,
    event_type  TEXT NOT NULL,
    payload     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_session_id ON events (session_id);

-- ── users ───────────────────────────────────────────────────────────
-- One row per app user. `tier` mirrors nullain-agent's existing
-- PlanTierLimits concept (agent/metering.py) — free/pro/enterprise.
CREATE TABLE IF NOT EXISTS users (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email       TEXT NOT NULL UNIQUE,
    tier        TEXT NOT NULL DEFAULT 'free' CHECK (tier IN ('free', 'pro', 'enterprise')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── sessions ────────────────────────────────────────────────────────
-- One row per conversation thread — the relational-side counterpart to
-- `events.session_id`, carrying fields useful for listing/filtering
-- sessions in a UI (which the append-only events table is not shaped
-- for). thread_id/tenant_id intentionally use the same string identity
-- as EventStorePort's session_id and nullain.rag's tenant_id — a
-- session and a RAG tenant scope are the same identifier space in this
-- schema, kept as plain TEXT (not a foreign key into `events`, which
-- has no unique constraint on session_id — one session has many rows).
CREATE TABLE IF NOT EXISTS sessions (
    thread_id       TEXT PRIMARY KEY,
    tenant_id       TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_active_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    model           TEXT,
    status          TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'archived', 'deleted'))
);

CREATE INDEX IF NOT EXISTS idx_sessions_tenant_id ON sessions (tenant_id);

-- ── metadata ────────────────────────────────────────────────────────
-- Generic per-session/per-user key-value store (user preferences, active
-- project, etc.) — the relational replacement for what
-- nullain-agent/agent/memory.py currently keeps in Redis + an in-process
-- dict. One row per (scope, scope_id, key) triple so both a session-scoped
-- and a user-scoped key can coexist without collision.
CREATE TABLE IF NOT EXISTS metadata (
    id          BIGSERIAL PRIMARY KEY,
    scope       TEXT NOT NULL CHECK (scope IN ('session', 'user')),
    scope_id    TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       JSONB NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (scope, scope_id, key)
);

CREATE INDEX IF NOT EXISTS idx_metadata_scope ON metadata (scope, scope_id);

-- ── traces ──────────────────────────────────────────────────────────
-- Structured execution log (tool calls, router decisions, errors) for
-- observability/analytics. A READ MODEL derived from the EventBus —
-- NEVER a parallel write to the same data `events` already owns as the
-- source of truth (see docs/FUSION_PLAN.md §4's explicit warning against
-- dual-write). The app subscribes to the EventBus and inserts rows here
-- as a side effect; if this table is lost or falls behind, replay from
-- `events` is unaffected.
CREATE TABLE IF NOT EXISTS traces (
    id          BIGSERIAL PRIMARY KEY,
    session_id  TEXT NOT NULL,
    trace_type  TEXT NOT NULL,  -- e.g. 'tool_call', 'router_decision', 'error'
    detail      JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_traces_session_id ON traces (session_id);
CREATE INDEX IF NOT EXISTS idx_traces_created_at ON traces (created_at);
