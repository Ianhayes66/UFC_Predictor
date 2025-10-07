# Migration Notes

The previous repository was an incomplete experiment and has been replaced with a fully structured implementation as mandated by the master prompt. Key changes:

- **Replaced** – Legacy `app/`, `model/`, and ad-hoc scripts were removed in favor of the standardized `src/ufc_winprob/` package.
- **Added** – Comprehensive pipelines, typed modules, tests, and documentation matching the prescribed tree.
- **Preserved** – Git history remains intact but prior assets are not referenced; run `scripts/download_sample_data.py` to bootstrap synthetic data compatible with the new schema.

Consumers of the old API should migrate to the FastAPI endpoints defined in `docs/API_SPEC.md`. Data artifacts now live in `data/processed/` and `data/models/` with Parquet storage replacing CSV.
