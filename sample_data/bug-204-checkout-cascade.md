# BUG-204: Checkout 5xx cascade during payment gateway latency spike

Reported by: bob
Severity: SEV-2
Component: checkout
Status: Fixed (see PR-482)

## Summary
On 2026-03-10 the payment gateway (Stripe) experienced a 4s p99 latency spike.
Because checkout calls the gateway synchronously, the checkout Tomcat thread
pool saturated at 200 threads and began rejecting requests. This propagated to
the storefront, which showed 5xx errors to 38% of users for 11 minutes.

## Root cause
Synchronous outbound call to the payment gateway on the request critical path
with no circuit breaker. The billing component's polling job also hammered the
checkout database, worsening contention.

## Fix
Decoupled via event bus (ADR-014). The payments-consumer microservice now
isolates gateway failures. A circuit breaker was added as defense in depth.
