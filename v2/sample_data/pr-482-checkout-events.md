# PR-482: Checkout event publisher + PaymentSettled consumer

Author: alice
Reviewed by: carol
Merged: 2026-03-18

## What
- Checkout microservice now publishes `OrderPlaced` to the event bus on order
  creation and consumes `PaymentSettled` to finalize the order.
- Added the `payments-consumer` microservice that wraps the payment gateway
  and emits `PaymentSettled` / `PaymentFailed`.
- Billing component updated to subscribe to `PaymentSettled` instead of polling
  the checkout database.

## Why
Implements ADR-014 to remove the synchronous payment gateway call from the
checkout critical path (root cause of BUG-204).

## Testing
- Integration test simulates a 10s gateway stall; checkout stays at p99 < 300ms.
- Contract test for `PaymentSettled` schema.

## Risk
Eventual consistency: a `PaymentSettled` that arrives after the order TTL is
reconciled by the nightly `RefundRequested` sweep.
