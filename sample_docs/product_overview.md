# Nimbus Cloud: Product Overview

Nimbus Cloud is a managed data-infrastructure platform for teams that need to
store, process, and ship data without operating their own servers. The
company was founded in 2019 and is headquartered in Austin, Texas, with a
secondary engineering office in Berlin, Germany.

## Core Products

- **Nimbus Compute** — serverless job execution for Python and SQL
  workloads. Jobs autoscale from zero to thousands of concurrent workers and
  bill per second of execution time.
- **Nimbus Storage** — an object storage service compatible with the S3 API,
  used for data lakes, backups, and static asset hosting.
- **Nimbus Pipelines** — a managed orchestration layer for building
  scheduled or event-triggered ETL/ELT workflows, with a visual DAG editor
  and a YAML-based pipeline definition format.
- **Nimbus Vault** — a secrets-management service for storing API keys,
  database credentials, and certificates, with automatic 90-day rotation for
  supported secret types.

## Who Uses Nimbus Cloud

Nimbus Cloud is aimed at small-to-midsize engineering teams (typically 5 to
500 engineers) that want production-grade data infrastructure without
hiring a dedicated platform team. Customers range from early-stage startups
running their first ETL pipeline to Series C companies migrating off
self-managed Kafka and Airflow clusters.

## Architecture at a Glance

All four products share a single control plane and a unified identity and
billing system, so a single Nimbus account and API key can provision
compute jobs, storage buckets, pipelines, and secrets. Nimbus Cloud runs on
top of two primary regions today: `us-east-1` and `eu-west-1`, with
additional regions on the public roadmap.

## Roadmap Highlights

The most frequently requested upcoming feature is Nimbus Streams, a managed
Kafka-compatible streaming product, currently in private beta with a small
group of design partners. General availability is targeted, but not
guaranteed, for the following fiscal year.
