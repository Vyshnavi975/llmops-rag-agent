# Nimbus Cloud: API Reference (Summary)

This document summarizes the Nimbus Cloud REST API for internal knowledge
base purposes. For the full OpenAPI specification, see the developer
portal.

## Base URL and Authentication

All API requests are made to `https://api.nimbuscloud.example/v1`. Requests
are authenticated with an API key passed in the `Authorization: Bearer
<key>` header. API keys are created and rotated from the Nimbus Vault
section of the dashboard and are scoped to a single workspace.

## Rate Limits

- **Starter plan**: 60 requests per minute per API key.
- **Team plan**: 1,000 requests per minute per API key.
- **Enterprise plan**: 5,000 requests per minute per API key, with the
  option to request a higher custom limit.

Requests beyond the limit receive an HTTP 429 response with a
`Retry-After` header indicating how many seconds to wait before retrying.

## Core Endpoints

- `POST /jobs` — submit a new Nimbus Compute job.
- `GET /jobs/{id}` — fetch the status and logs of a job.
- `POST /buckets` — create a new Nimbus Storage bucket.
- `POST /pipelines/{id}/trigger` — manually trigger a Nimbus Pipelines run
  outside of its normal schedule.
- `GET /secrets` — list secret names in Nimbus Vault (values are never
  returned by the API; they can only be injected into a running job).

## Webhooks

Nimbus Cloud can send webhook notifications for the `job.completed`,
`job.failed`, and `pipeline.run.completed` events. Webhook endpoints must
respond with a 2xx status code within 5 seconds, or the delivery is retried
up to 3 times with exponential backoff.

## SDKs

Official SDKs are published for Python and JavaScript/TypeScript. Both
SDKs wrap the REST API and handle authentication, retries, and pagination
automatically. A community-maintained Go SDK also exists but is not
officially supported by Nimbus Cloud.
