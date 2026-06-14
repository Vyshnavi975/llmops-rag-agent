# Nimbus Cloud: Incident Response Runbook

This runbook describes how the Nimbus Cloud engineering team handles
production incidents. It is included in the knowledge base because support
and success teams frequently need to answer customer questions about
incident handling and uptime commitments.

## On-Call Rotation

Nimbus Cloud engineering runs a weekly on-call rotation managed through
PagerDuty. Each region (`us-east-1` and `eu-west-1`) has its own primary
and secondary on-call engineer. Escalation to a secondary on-call engineer
happens automatically if the primary does not acknowledge a page within 10
minutes.

## Severity Levels

- **SEV1** — full or near-full outage of a core product (Nimbus Compute,
  Storage, Pipelines, or Vault) affecting many customers. Must be
  acknowledged within **5 minutes** and a status page update posted within
  15 minutes.
- **SEV2** — significant degradation affecting a subset of customers or a
  single region. Must be acknowledged within 15 minutes.
- **SEV3** — minor issue with a workaround available, or an issue affecting
  a single customer. Handled during normal business hours.

## Status Page and Communication

Live incident status is published at `status.nimbuscloud.example`.
Enterprise customers with a Premium support plan also receive proactive
notifications in their shared Slack channel for any SEV1 or SEV2 incident
affecting their workspace's region.

## Postmortems

Every SEV1 incident receives a public postmortem published within **3
business days** of resolution, including a timeline, root cause, and
follow-up action items. SEV2 postmortems are internal-only unless a
customer specifically requests a copy.

## Uptime Commitment

Nimbus Cloud targets **99.9% monthly uptime** for the core API and control
plane on the Team and Enterprise plans. Enterprise customers with a
negotiated SLA may receive service credits for uptime below the committed
threshold, as defined in their contract.
