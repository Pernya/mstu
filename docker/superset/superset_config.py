from __future__ import annotations

import os


SECRET_KEY = os.getenv("SUPERSET_SECRET_KEY", "retail-superset-secret-key-change-in-production")
SQLALCHEMY_DATABASE_URI = os.getenv("SUPERSET_METADATA_DB_URI", "sqlite:////app/superset_home/superset.db")
FEATURE_FLAGS = {"ALERT_REPORTS": False}
WTF_CSRF_ENABLED = True
