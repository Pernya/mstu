from __future__ import annotations

import os

from superset.app import create_app
from superset.extensions import db
from superset.models.core import Database


def register_database() -> None:
    """Регистрирует подключение Superset к витринам PostgreSQL через host.docker.internal."""
    uri = os.getenv("SUPERSET_RETAIL_DATABASE_URI")
    if not uri:
        return
    app = create_app()
    with app.app_context():
        database = db.session.query(Database).filter_by(database_name="Retail DWH").one_or_none()
        if database is None:
            database = Database(database_name="Retail DWH", expose_in_sqllab=True)
            db.session.add(database)
        database.sqlalchemy_uri = uri
        db.session.commit()


if __name__ == "__main__":
    register_database()
