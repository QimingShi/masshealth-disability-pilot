"""Local-override template for the Postgres connection.

HOW TO USE
----------
1. Copy this file:
       cp db/local_override.example.py db/local_override.py
       (or in cmd:  copy db\local_override.example.py db\local_override.py)
2. Edit db/local_override.py — replace <PUT_YOUR_PASSWORD_HERE> with the real password.
3. That's it. pipeline/db.py.connect_pg() automatically picks up the override.

THE OVERRIDE IS GITIGNORED. db/local_override.py never gets committed to
either remote — it lives only on your machine.

This pattern lets you keep credentials in a SINGLE local file rather than
juggling env vars across cmd / PowerShell / Python interpreters. It's the
same pattern Django uses (local_settings.py) and many other frameworks.

IMPORTANT: the password in db/local_override.py is still in cleartext on
your disk. Treat the file accordingly:
  - File permissions: don't share write/read with other users on the machine
  - When you no longer need it, delete the file
  - Rotate the DB password when you suspect it's been exposed
"""
import psycopg2
from urllib.parse import urlparse, unquote


def connect_pg():
    """Hardcoded local connection — for dev/test only."""
    DATABASE_URL = (
        "postgresql://shiq:<PUT_YOUR_PASSWORD_HERE>"
        "@expedite-nonprod-rds.cfqvorcy6lau.us-east-1.rds.amazonaws.com:5432/expedite"
    )

    parsed = urlparse(DATABASE_URL)
    return psycopg2.connect(
        host=parsed.hostname,
        port=parsed.port or 5432,
        user=unquote(parsed.username),
        password=unquote(parsed.password),
        dbname=parsed.path.lstrip("/"),
        sslmode="require",   # RDS requires SSL; without this you get a refused connection
    )
