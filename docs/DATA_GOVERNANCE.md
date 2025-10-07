# Data Governance

- **Lineage** – Each pipeline stage writes deterministic artifacts to `data/` with timestamps. Metadata columns capture source and processing context.
- **Retention** – Raw HTML snapshots kept for 30 days; processed Parquet files versioned by run. Use `make clean` for local pruning.
- **Access Control** – Default storage backend is Parquet; Postgres schema defined in code for optional deployment. Apply RBAC when deploying to managed databases.
- **Quality** – Great Expectations suites can be layered; current implementation validates schema shapes via tests. On failures, pipelines abort before publishing predictions.
