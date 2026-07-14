# Runbook: checkout service load test

## Scope

Validate the checkout service under peak traffic before each release. Covers the
order write path, the payments call (now async per PR-482), and PostgreSQL
primary/replica health.

## Steps

1. Ramp traffic to 3x Black-Friday peak against staging.
2. Watch the connection pool on the checkout service; alert if >80% used.
3. Inject 200ms latency on the PostgreSQL replica; confirm orders still complete
   via the `pending_payment` reconcile path (no cascade).
4. Confirm the payments service statement timeout (5s via PgBouncer) holds.

## Owners

- checkout service: dave
- PostgreSQL: carol
- on-call: bob

## Notes

This runbook was written after BUG-204 to prevent regression of the cascade
timeout. It exercises the same path PR-482 hardened.
