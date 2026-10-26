"""Irreversible training reservations with a single active worker."""

import json
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path

LIMITS = {"search": 11, "probe": 2, "audit": 1, "final": 1}


class BudgetError(RuntimeError):
    pass


class Ledger:
    def __init__(self, path, *, dataset_hash, split_hash):
        self.path = Path(path).resolve()
        try:
            with self._db() as db:
                if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise ValueError("Corrupt ledger")
                meta = json.loads(db.execute("SELECT value FROM metadata WHERE key='identity'")
                                  .fetchone()[0])
        except (sqlite3.Error, TypeError, OSError, json.JSONDecodeError) as error:
            raise ValueError("Missing or corrupt ledger; refusing to reset campaign") from error
        if meta["dataset_hash"] != dataset_hash or meta["split_hash"] != split_hash:
            raise ValueError("Campaign fingerprint changed")
        self.identity = meta

    @classmethod
    def initialize(cls, path, *, dataset_hash, split_hash, kind="real"):
        if kind not in {"real", "simulation"}:
            raise ValueError("Unknown campaign kind")
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.open("xb").close()
        except FileExistsError as error:
            raise ValueError("Campaign already exists; load it instead of resetting") from error
        with closing(sqlite3.connect(path)) as db, db:
            db.executescript("""
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    phase TEXT NOT NULL, recipe_key TEXT NOT NULL, split_hash TEXT NOT NULL,
                    status TEXT NOT NULL, created TEXT NOT NULL, pid INTEGER,
                    details TEXT NOT NULL, result TEXT, error TEXT,
                    UNIQUE(phase, recipe_key, split_hash)
                );
            """)
            identity = {"dataset_hash": dataset_hash, "split_hash": split_hash,
                        "kind": kind, "budget": 15, "schema_version": 1}
            db.execute("INSERT INTO metadata VALUES ('identity', ?)", (json.dumps(identity),))
        return cls(path, dataset_hash=dataset_hash, split_hash=split_hash)

    @contextmanager
    def _db(self):
        db = sqlite3.connect(f"file:{self.path.as_posix()}?mode=rw", uri=True, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @property
    def used(self):
        with self._db() as db:
            return db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]

    @property
    def remaining(self):
        return 15 - self.used

    @property
    def audit_lock(self):
        with self._db() as db:
            row = db.execute("SELECT value FROM metadata WHERE key='audit_lock'").fetchone()
            return json.loads(row[0]) if row else None

    def reserve(self, phase, recipe_key, split_hash, *, details=None):
        if phase not in LIMITS:
            raise ValueError("Unknown attempt phase")
        payload = json.dumps(details or {}, sort_keys=True, allow_nan=False)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM metadata WHERE key='audit_lock'").fetchone():
                raise BudgetError("Training prohibited after audit lock")
            if db.execute("SELECT 1 FROM attempts WHERE status IN ('reserved','running')").fetchone():
                raise BudgetError("An active worker/reservation must be reconciled first")
            used = db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
            count = db.execute("SELECT COUNT(*) FROM attempts WHERE phase=?", (phase,)).fetchone()[0]
            if used >= 15 or count >= LIMITS[phase]:
                raise BudgetError("Attempt or phase budget exhausted")
            if phase in {"search", "probe"}:
                terminal = db.execute("SELECT 1 FROM attempts WHERE phase IN ('audit','final')")
                if used >= 13 or terminal.fetchone():
                    raise BudgetError("Two terminal tokens are reserved; search is closed")
            if phase == "final" and not db.execute(
                    "SELECT 1 FROM attempts WHERE phase='audit'").fetchone():
                raise BudgetError("The audit-stage attempt must precede final fitting")
            try:
                cursor = db.execute(
                    "INSERT INTO attempts(phase,recipe_key,split_hash,status,created,details) "
                    "VALUES (?,?,?,'reserved',?,?)",
                    (phase, recipe_key, split_hash, datetime.now(timezone.utc).isoformat(), payload),
                )
            except sqlite3.IntegrityError as error:
                raise BudgetError("Recipe already attempted; automatic retries are prohibited") from error
            return cursor.lastrowid

    def mark_running(self, attempt_id, pid):
        with self._db() as db:
            cursor = db.execute("UPDATE attempts SET status='running',pid=? "
                                "WHERE id=? AND status='reserved'", (pid, attempt_id))
            if cursor.rowcount != 1:
                raise ValueError("Attempt is not reserved")

    def finish(self, attempt_id, *, success, result=None, error=None):
        payload = json.dumps(result or {}, sort_keys=True, allow_nan=False)
        with self._db() as db:
            cursor = db.execute("UPDATE attempts SET status=?,result=?,error=? "
                                "WHERE id=? AND status IN ('reserved','running')",
                                ("succeeded" if success else "failed", payload, error, attempt_id))
            if cursor.rowcount != 1:
                raise ValueError("Attempt is not active; results cannot be overwritten")

    def attempts(self):
        with self._db() as db:
            rows = [dict(row) for row in db.execute("SELECT * FROM attempts ORDER BY id")]
        for row in rows:
            row["details"] = json.loads(row["details"])
            row["result"] = json.loads(row["result"]) if row["result"] else None
        return rows

    def lock_audit(self, identities):
        if set(identities) != {"submission", "proxy", "audit_predictions"}:
            raise ValueError("Incomplete audit identities")
        payload = json.dumps(identities, sort_keys=True, allow_nan=False)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute("SELECT value FROM metadata WHERE key='audit_lock'").fetchone()
            if previous:
                if json.loads(previous[0]) != identities:
                    raise ValueError("Audit identities are already locked")
                return
            if db.execute("SELECT 1 FROM attempts WHERE status IN ('reserved','running')").fetchone():
                raise BudgetError("Cannot unlock audit while a worker is active")
            db.execute("INSERT INTO metadata VALUES ('audit_lock', ?)", (payload,))

    def reconcile_interrupted(self, *, confirm_no_active_worker=False):
        if not confirm_no_active_worker:
            raise ValueError("Verify the recorded worker has stopped before reconciliation")
        with self._db() as db:
            db.execute("UPDATE attempts SET status='interrupted',error='Operator reconciled worker' "
                       "WHERE status IN ('reserved','running')")
