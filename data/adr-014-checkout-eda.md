# ADR-014: Migrate Checkout to Event-Driven Architecture

Status: Accepted
Author: alice
Date: 2026-03-12

## Context
The checkout service currently calls the payment gateway synchronously. A
latency spike in the payment gateway directly blocks the checkout request
thread pool, causing a cascade of 5xx errors across the storefront. The billing
component depends on the checkout service for invoice generation.

## Decision
We will decouple checkout from the payment gateway using an event bus.
The checkout microservice publishes `OrderPlaced` and consumes
`PaymentSettled`. The payment gateway becomes a downstream consumer instead of
a synchronous dependency.

## Consequences
- The payment gateway no longer impacts checkout availability.
- The billing component must subscribe to `PaymentSettled` instead of polling
  the checkout database.
- A new `payments-consumer` microservice is introduced to isolate gateway
  failures.
- Tradeoff: eventual consistency; refunds require a compensating `RefundRequested`
  event.

## Related
See PR-482 for the checkout publisher and BUG-204 for the original latency
incident.
