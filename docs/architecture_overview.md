# Cloud Architecture & Failover Specification

## 1. High Availability & Failover Policy
All primary transactional databases run on a active-passive cluster with synchronous streaming replication. 
If the primary node experiences network isolation or hardware failure, the automated failover coordinator promotes the standby replica in under 5 seconds with zero data loss (RPO = 0, RTO < 5s).

## 2. API Quotas and Rate Limits
- Public REST endpoints are rate-limited to 100 requests per second per IP address.
- Enterprise customers receive a guaranteed throughput tier of 1,000 requests per second with burst allowance up to 2,500 requests per second.
- Rate limit violations return HTTP 429 Too Many Requests with a `Retry-After` header.

## 3. Authentication & Concurrency
The authentication service issues short-lived JWT access tokens with a 15-minute validity window and refresh tokens with a 7-day sliding window. Token refresh operations are guarded by a distributed mutex lock to prevent concurrent worker race conditions.
