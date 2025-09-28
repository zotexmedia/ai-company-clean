-- Enable UUID generation helpers
create extension if not exists "uuid-ossp";

-- Enum to track job status lifecycle
do $$
begin
    if not exists (select 1 from pg_type where typname = 'job_status') then
        create type job_status as enum ('queued', 'running', 'done', 'failed', 'partial');
    end if;
end $$;

create table if not exists canonical_company (
    id uuid primary key default uuid_generate_v4(),
    canonical_name text not null unique,
    key_form text not null unique,
    first_seen timestamptz not null default timezone('utc', now()),
    last_seen timestamptz not null default timezone('utc', now()),
    confidence_avg real not null default 0,
    aliases_count integer not null default 0
);

create table if not exists alias (
    id uuid primary key default uuid_generate_v4(),
    alias_name text not null,
    canonical_id uuid not null references canonical_company(id) on delete cascade,
    source text,
    first_seen timestamptz not null default timezone('utc', now()),
    last_seen timestamptz not null default timezone('utc', now()),
    confidence_last real not null default 0,
    details jsonb,
    unique (alias_name, canonical_id)
);

create index if not exists alias_canonical_id_idx on alias (canonical_id);
create index if not exists alias_alias_name_idx on alias using gin (to_tsvector('english', alias_name));

create table if not exists job_run (
    id uuid primary key default uuid_generate_v4(),
    status job_status not null default 'queued',
    input_count integer not null default 0,
    success_count integer not null default 0,
    error_count integer not null default 0,
    created_at timestamptz not null default timezone('utc', now()),
    updated_at timestamptz not null default timezone('utc', now()),
    result_path text
);

create index if not exists job_run_status_idx on job_run (status);
