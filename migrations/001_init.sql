-- SpireDossier schema. Run once in the Supabase SQL editor.
--
-- Design note: roles, applicants and dossiers keep their payload in jsonb
-- rather than a column per field. The Pydantic models are the source of truth
-- and still moving; mirroring them in DDL would mean a migration every time a
-- field is added, for no benefit at this scale. The columns that exist are the
-- ones actually filtered on -- company_id, email -- and those are indexed.
--
-- Row-level security is ON with no public policies, so the anon key can read
-- nothing here. These tables hold candidate names, contact details, salary
-- expectations and full CV text; the application reaches them with the
-- service_role key from the server only.

create extension if not exists "pgcrypto";

-- ---------------------------------------------------------------- companies
create table if not exists public.companies (
    id            text primary key,
    name          text not null,
    email         text not null unique,
    password_hash text not null,
    is_admin      boolean not null default false,
    industry      text,
    location      text,
    website       text,
    created_at    timestamptz not null default now()
);

create index if not exists companies_email_idx on public.companies (lower(email));

-- ---------------------------------------------------------------- roles
create table if not exists public.roles (
    id         text primary key,
    company_id text references public.companies (id) on delete cascade,
    payload    jsonb not null,
    created_at timestamptz not null default now()
);

create index if not exists roles_company_idx on public.roles (company_id);

-- ---------------------------------------------------------------- applicants
create table if not exists public.applicants (
    id          text primary key,
    filename    text not null,
    prefs       jsonb not null,
    -- Where the CV actually lives. storage_key points at the resumes bucket;
    -- local_path is the working copy and is not guaranteed to survive a
    -- redeploy, which is exactly why storage_key exists.
    storage_key text,
    local_path  text,
    created_at  timestamptz not null default now()
);

-- ---------------------------------------------------------------- dossiers
-- Only what cannot be recomputed is stored. The timeline, the computed risk
-- flags and the quote verification are all derived deterministically from the
-- profile and the CV text, so they are rebuilt on read rather than persisted
-- and allowed to drift from the code that produces them.
create table if not exists public.dossiers (
    id          text primary key,
    profile     jsonb not null,
    brief       jsonb not null,
    assessment  jsonb not null,
    document    jsonb not null,
    brief_text  text not null default '',
    model       text,
    anonymise   boolean not null default true,
    usage       jsonb,
    elapsed     real,
    warnings    jsonb,
    created_at  timestamptz not null default now()
);

create index if not exists dossiers_created_idx on public.dossiers (created_at desc);

-- ---------------------------------------------------------------- lockdown
alter table public.companies  enable row level security;
alter table public.roles      enable row level security;
alter table public.applicants enable row level security;
alter table public.dossiers   enable row level security;

-- Deliberately no policies. Without one, anon and authenticated roles are
-- denied everything; service_role bypasses RLS and is what the server uses.
