# Wiki: Payments Platform Topology

Owner: platform-team

The payments platform consists of these microservices:

- **checkout** — orchestrates the order. Publishes `OrderPlaced`, consumes
  `PaymentSettled`. Depends on the event bus and the billing component.
- **payments-consumer** — isolates the payment gateway (Stripe). Translates
  gateway webhooks into `PaymentSettled` / `PaymentFailed` domain events.
- **billing** — generates invoices. Subscribes to `PaymentSettled`. Previously
  polled the checkout database directly, which caused DB contention (BUG-204).
- **payment gateway (Stripe)** — external dependency. A latency spike here used
  to cascade into checkout (see BUG-204); now isolated behind payments-consumer.

## Failure modes
If the payment gateway degrades, only `payments-consumer` is affected. Checkout
and the storefront remain available because they no longer call the gateway
synchronously (ADR-014, PR-482).

## On-call runbook
1. Check payments-consumer error rate.
2. Verify `PaymentSettled` is flowing on the event bus.
3. If stalled, replay the dead-letter queue.
