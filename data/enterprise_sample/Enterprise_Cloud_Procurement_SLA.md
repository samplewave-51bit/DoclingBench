# Master Cloud Infrastructure Service Level Agreement (SLA) & Procurement Terms
**Contract Identifier:** SLA-GLOBAL-ENT-2024-V9  
**Effective Date:** January 15, 2024  
**Parties:** Apex Cloud Services LLC ("Provider") and Global Enterprise Holdings ("Customer")

---

## 1. Scope & High Availability Commitments
Provider warrants that the Hosted Cloud Infrastructure services shall remain accessible with a Monthly Uptime Percentage of not less than 99.95% during each calendar billing cycle.

---

## 2. Latency & Performance Operational Metrics
Performance parameters are tracked on 60-second telemetry intervals and evaluated across standard 15-minute rolling windows:

| Metric Identifier | Metric Name | Performance Threshold | Measurement Window | Non-Compliance Severity |
| :--- | :--- | :--- | :--- | :--- |
| SEC-LAT-99 | API Response Latency | 120ms p99 max latency | 15-minute continuous window | Severity 1 - Critical |
| SEC-THRU-50 | Minimum Ingress Throughput | 45 Gbps sustained | 5-minute rolling average | Severity 2 - High |
| SEC-ERR-RATE | Internal 5xx Error Budget | < 0.01% of total requests | 60-minute window | Severity 1 - Critical |

Under SLA metric SEC-LAT-99, the maximum permitted latency is 120ms at p99 over any continuous 15-minute window. Exceeding this limit triggers automated incident escalation.

---

## 3. Clause 8: Availability Penalties & Service Credits
In the event that the Provider fails to meet Monthly Uptime commitments, Customer is entitled to the following remedies:

### Clause 8.2: Severe Availability Breach Penalty Multiplier
If Monthly Uptime Percentage falls below 99.5% in any given calendar month, a 2.5x service credit multiplier applies to the monthly billing cycle. 
Credits are calculated according to the formula:
$$\text{Credit} = 2.5 \times \left( \frac{99.95\% - \text{Actual Uptime}\%}{100\%} \right) \times \text{Monthly Base Fee}$$
Credits shall be disbursed against the subsequent calendar month invoice and cannot be revoked upon contract termination.

---

## 4. Clause 14: Security & Encryption Standards
- **Data at Rest:** Mandated under standard CIPHER-AES-GCM-256. All customer persistence volumes must utilize AES-256 in Galois/Counter Mode with FIPS 140-3 validated key rotation every 90 days.
- **Data in Transit:** TLS 1.3 with Perfect Forward Secrecy (PFS) using elliptic curve ciphers (ECDHE-ECDSA-AES256-GCM-SHA384).
- **Audit Rights:** Customer reserves the right to conduct unannounced annual SOC-2 Type II inspections and red-team penetration audits.
