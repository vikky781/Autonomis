"""Irreversible training reservations with a single active worker."""

import hashlib
import json
import sqlite3
from contextlib import closing, contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path

LIMITS = {"search": 11, "probe": 2, "audit": 1, "final": 1}


class BudgetError(RuntimeError):
    pass


class Ledger:
    def __init__(self, path, *, dataset_hash, split_hash, history_ledgers=()):
        self.path = Path(path).resolve()
        self.history_ledgers = tuple(Path(item).resolve() for item in history_ledgers)
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
        self.budget_path = self._budget_path(self.path)
        if self.identity["kind"] == "real":
            self._ensure_budget_store()

    @staticmethod
    def _budget_path(path):
        """Return the one append-only budget index shared by sibling campaigns."""
        path = Path(path).resolve()
        scope = next((parent for parent in path.parents if parent.name == "campaigns"),
                     path.parent)
        return scope / ".global-budget.sqlite"

    @contextmanager
    def _budget_db(self):
        self.budget_path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.budget_path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _event_key(row):
        identity = {key: row[key] for key in
                    ("phase", "recipe_key", "split_hash", "created", "details")}
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def _campaign_ledgers(self):
        """Find ledgers in this campaign scope, including archived campaign folders."""
        scope = self.budget_path.parent
        paths = set(scope.rglob("ledger.sqlite")) if scope.name == "campaigns" else set()
        paths.add(self.path)
        return sorted(path.resolve() for path in paths if path.resolve() != self.budget_path)

    def _ensure_budget_store(self):
        with self._budget_db() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("""CREATE TABLE IF NOT EXISTS attempts (
                event_key TEXT PRIMARY KEY, campaign_path TEXT NOT NULL,
                local_id INTEGER NOT NULL, phase TEXT NOT NULL, status TEXT NOT NULL,
                created TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY, value TEXT NOT NULL)""")
            self._sync_budget(db)

    def _sync_budget(self, budget):
        """Reconcile local and archived ledgers before computing the global allowance."""
        for path in self._campaign_ledgers():
            if not path.is_file():
                continue
            try:
                with closing(sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)) as source:
                    source.row_factory = sqlite3.Row
                    identity_row = source.execute(
                        "SELECT value FROM metadata WHERE key='identity'").fetchone()
                    if not identity_row or json.loads(identity_row[0]).get("kind") != "real":
                        continue
                    for row in source.execute("SELECT * FROM attempts ORDER BY id"):
                        key = self._event_key(row)
                        budget.execute("INSERT OR IGNORE INTO attempts VALUES (?,?,?,?,?,?)",
                            (key, str(path), row["id"], row["phase"], row["status"], row["created"]))
                        budget.execute("UPDATE attempts SET status=? WHERE event_key=?",
                                       (row["status"], key))
                    lock = source.execute(
                        "SELECT value FROM metadata WHERE key='audit_lock'").fetchone()
                    if lock:
                        current = budget.execute(
                            "SELECT value FROM metadata WHERE key='audit_lock'").fetchone()
                        if current and json.loads(current[0]) != json.loads(lock[0]):
                            raise ValueError("Conflicting Audit locks across campaign ledgers")
                        budget.execute("INSERT OR IGNORE INTO metadata VALUES ('audit_lock', ?)",
                                       (lock[0],))
            except sqlite3.Error as error:
                raise ValueError(f"Cannot reconcile campaign ledger {path}") from error

        for path in self.history_ledgers:
            if not path.is_file():
                raise ValueError(f"Historical attempt ledger is missing; refusing new training: {path}")
            try:
                with path.open(encoding="utf-8") as source:
                    for line_number, line in enumerate(source, start=1):
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        run_id = row.get("run_id")
                        if not run_id:
                            raise ValueError(f"Missing run_id in historical ledger {path}:{line_number}")
                        fingerprint = hashlib.sha256(json.dumps({
                            key: row.get(key) for key in
                            ("run_id", "started_at", "dataset_sha256", "config")
                        }, sort_keys=True).encode()).hexdigest()
                        legacy_key = f"legacy:{run_id}"
                        previous = budget.execute(
                            "SELECT campaign_path FROM attempts WHERE event_key=?", (legacy_key,)
                        ).fetchone()
                        if previous and previous["campaign_path"].startswith("legacy:"):
                            known = previous["campaign_path"].split(":", 1)[1]
                            if known != fingerprint:
                                raise ValueError(f"Conflicting historical records for run_id {run_id}")
                        phase = row.get("config", {}).get("experiment", {}).get("stage") or "legacy"
                        budget.execute("INSERT OR IGNORE INTO attempts VALUES (?,?,?,?,?,?)",
                            (legacy_key, f"legacy:{fingerprint}", int(row.get("seq", line_number)),
                             phase, row.get("status", "unknown"),
                             row.get("started_at") or "unknown"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"Cannot read historical attempt ledger {path}") from error

    @classmethod
    def initialize(cls, path, *, dataset_hash, split_hash, kind="real", history_ledgers=()):
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
        return cls(path, dataset_hash=dataset_hash, split_hash=split_hash,
                   history_ledgers=history_ledgers)

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
        if self.identity["kind"] == "real":
            with self._budget_db() as db:
                db.execute("BEGIN IMMEDIATE")
                self._sync_budget(db)
                return db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
        with self._db() as db:
            return db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]

    @property
    def remaining(self):
        return max(0, 15 - self.used)

    @property
    def audit_lock(self):
        if self.identity["kind"] == "real":
            with self._budget_db() as db:
                db.execute("BEGIN IMMEDIATE")
                self._sync_budget(db)
                row = db.execute("SELECT value FROM metadata WHERE key='audit_lock'").fetchone()
                return json.loads(row[0]) if row else None
        with self._db() as db:
            row = db.execute("SELECT value FROM metadata WHERE key='audit_lock'").fetchone()
            return json.loads(row[0]) if row else None

    def reserve(self, phase, recipe_key, split_hash, *, details=None):
        if phase not in LIMITS:
            raise ValueError("Unknown attempt phase")
        payload = json.dumps(details or {}, sort_keys=True, allow_nan=False)
        budget_context = self._budget_db() if self.identity["kind"] == "real" else nullcontext(None)
        with budget_context as budget:
            if budget is not None:
                budget.execute("BEGIN IMMEDIATE")
                self._sync_budget(budget)
                if budget.execute("SELECT 1 FROM metadata WHERE key='audit_lock'").fetchone():
                    raise BudgetError("Training prohibited after audit lock")
                if budget.execute("SELECT 1 FROM attempts WHERE status IN ('reserved','running')").fetchone():
                    raise BudgetError("An active worker/reservation exists in another campaign")
                global_used = budget.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
                global_phase_count = budget.execute(
                    "SELECT COUNT(*) FROM attempts WHERE phase=?", (phase,)).fetchone()[0]
                if global_used >= 15 or global_phase_count >= LIMITS[phase]:
                    raise BudgetError("Global attempt or phase budget exhausted across campaigns")
                if phase in {"search", "probe"}:
                    if global_used >= 13 or budget.execute(
                            "SELECT 1 FROM attempts WHERE phase IN ('audit','final')").fetchone():
                        raise BudgetError("Two terminal tokens are reserved; search is closed")
                if phase == "final" and not budget.execute(
                        "SELECT 1 FROM attempts WHERE phase='audit'").fetchone():
                    raise BudgetError("The global audit-stage attempt must precede final fitting")
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
                if budget is not None:
                    row = db.execute("SELECT * FROM attempts WHERE id=?", (cursor.lastrowid,)).fetchone()
                    budget.execute("INSERT OR IGNORE INTO attempts VALUES (?,?,?,?,?,?)",
                                   (self._event_key(row), str(self.path), row["id"], row["phase"],
                                    row["status"], row["created"]))
                return cursor.lastrowid

    def mark_running(self, attempt_id, pid):
        with self._db() as db:
            cursor = db.execute("UPDATE attempts SET status='running',pid=? "
                                "WHERE id=? AND status='reserved'", (pid, attempt_id))
            if cursor.rowcount != 1:
                raise ValueError("Attempt is not reserved")
        self._sync_current_attempt(attempt_id)

    def finish(self, attempt_id, *, success, result=None, error=None):
        payload = json.dumps(result or {}, sort_keys=True, allow_nan=False)
        with self._db() as db:
            cursor = db.execute("UPDATE attempts SET status=?,result=?,error=? "
                                "WHERE id=? AND status IN ('reserved','running')",
                                ("succeeded" if success else "failed", payload, error, attempt_id))
            if cursor.rowcount != 1:
                raise ValueError("Attempt is not active; results cannot be overwritten")
        self._sync_current_attempt(attempt_id)

    def _sync_current_attempt(self, attempt_id):
        if self.identity["kind"] != "real":
            return
        with self._db() as db:
            row = db.execute("SELECT * FROM attempts WHERE id=?", (attempt_id,)).fetchone()
        with self._budget_db() as budget:
            budget.execute("BEGIN IMMEDIATE")
            key = self._event_key(row)
            budget.execute("UPDATE attempts SET status=? WHERE event_key=?", (row["status"], key))

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
        budget_context = self._budget_db() if self.identity["kind"] == "real" else nullcontext(None)
        with budget_context as budget:
            if budget is not None:
                budget.execute("BEGIN IMMEDIATE")
                self._sync_budget(budget)
                current = budget.execute("SELECT value FROM metadata WHERE key='audit_lock'").fetchone()
                if current and json.loads(current[0]) != identities:
                    raise ValueError("Audit identities are already locked across campaigns")
                if budget.execute(
                        "SELECT 1 FROM attempts WHERE status IN ('reserved','running')").fetchone():
                    raise BudgetError("Cannot lock audit while a worker is active in any campaign")
                budget.execute("INSERT OR IGNORE INTO metadata VALUES ('audit_lock', ?)", (payload,))
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
        if self.identity["kind"] == "real":
            with self._budget_db() as budget:
                budget.execute("BEGIN IMMEDIATE")
                self._sync_budget(budget)
