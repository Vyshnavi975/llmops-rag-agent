# Nimbus Cloud: Security and Compliance

Security is treated as a product feature at Nimbus Cloud, not an
afterthought. This document summarizes the controls most frequently asked
about by prospective customers' security teams.

## Certifications

Nimbus Cloud has completed a **SOC 2 Type II** audit and can provide the
report to customers under NDA. Nimbus Cloud is also **GDPR compliant** for
customers processing personal data of EU residents, and offers a signed
**Business Associate Agreement (BAA)** to Enterprise-tier customers who need
HIPAA-eligible workloads. Nimbus Cloud is not currently FedRAMP authorized.

## Encryption

All customer data stored in Nimbus Storage and Nimbus Vault is encrypted at
rest using **AES-256**. Data in transit between customer systems and the
Nimbus Cloud API, as well as between internal Nimbus services, is encrypted
using **TLS 1.3**. Secrets stored in Nimbus Vault are additionally
encrypted with a per-customer envelope key.

## Data Residency

Customer data is stored in the region selected at account setup, currently
either `us-east-1` (Virginia, USA) or `eu-west-1` (Dublin, Ireland).
Cross-region replication is available on the Enterprise plan for disaster
recovery purposes but is off by default.

## Access Control

Nimbus Cloud supports role-based access control (RBAC) with four built-in
roles: Viewer, Editor, Admin, and Billing. Enterprise-tier accounts can also
configure single sign-on (SSO) via SAML 2.0 and enforce multi-factor
authentication (MFA) for all workspace members.

## Vulnerability Management

Nimbus Cloud runs continuous automated vulnerability scanning against its
infrastructure and conducts an external penetration test at least once per
year. Critical vulnerabilities identified internally are patched within 72
hours of confirmation. Nimbus Cloud also operates a public bug-bounty
program for responsible disclosure of security issues.
