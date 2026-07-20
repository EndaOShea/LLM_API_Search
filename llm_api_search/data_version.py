# Rewritten by scripts/update_models.py on each refresh run — records the date
# the static model data was last refreshed from live provider APIs. Consumed
# by catalog.py as the /catalog.json "generated_at" stamp so downstream sync
# scripts can detect a stale upstream.
DATA_UPDATED = "2026-07-20"
