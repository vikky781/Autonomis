"""Loopback HTTP server for the campaign dashboard and explicit run controls."""

import ipaddress
import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from autonomis.campaign import run_campaign, run_single_search_attempt, start_new_round
from autonomis.reporting import render_report
from autonomis.storage import read_json


def serve_report(ledger, output_dir, *, dataset_role, config=None, host="127.0.0.1", port=8502):
    """Serve the campaign demo and explicitly requested training actions on loopback."""
    output_dir = Path(output_dir)
    report_path = output_dir / "report.html"
    summary_path = output_dir / "summary.json"
    csrf_token = secrets.token_urlsafe(32)
    try:
        is_loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        is_loopback = host.lower() == "localhost"
    operation_lock = threading.Lock()
    operation = {"id": None, "state": "idle", "message": "No training action started."}

    def control_status():
        reason = ""
        enabled = config is not None
        if config is None:
            reason = "Training actions are not configured for this server."
        else:
            try:
                config.permit.check()
            except Exception as error:
                enabled = False
                reason = str(error)
            if ledger.identity.get("kind") != "real":
                enabled = False
                reason = "Training controls are unavailable for simulation ledgers."
            if not is_loopback:
                enabled = False
                reason = "Training controls are disabled unless the server binds to a loopback address."
        audit_locked = ledger.audit_lock is not None
        with operation_lock:
            current = dict(operation)
        busy = current["state"] == "running"
        campaign_rows = ledger.attempts()
        phase_counts = ledger.phase_counts()
        attempts_used = ledger.used
        remaining = ledger.remaining
        training_enabled = enabled and not audit_locked
        manual_enabled = training_enabled and not busy and remaining > 2
        if manual_enabled and phase_counts.get("search", 0) >= 11:
            manual_enabled = False
        if manual_enabled and ((output_dir / "selection.json").exists()
                               or (output_dir / "frozen-recipe.json").exists()):
            manual_enabled = False
        local_search_count = phase_counts.get("search", 0)
        can_finish = (local_search_count > 0 or (output_dir / "frozen-recipe.json").exists()
                      or (output_dir / "selection.json").exists())
        full_enabled = (training_enabled and not busy and remaining > 0
                        and (ledger.remaining > 2 or can_finish))
        if full_enabled and (output_dir / "selection.json").exists():
            full_enabled = False
        if busy and not reason:
            reason = "A training action is already running."
        if audit_locked and not reason:
            reason = "This round's Audit is locked. Archive it to history and start a new round to use remaining attempts."
        if (enabled and not busy and not audit_locked and not full_enabled
                and ledger.remaining <= 2):
            reason = ("This round has no Search candidate, and the remaining terminal tokens are reserved. "
                      "A new round needs at least three shared attempts.")
        return {"enabled": enabled, "manual_enabled": manual_enabled,
                "full_enabled": full_enabled, "reason": reason,
                "audit_locked": audit_locked,
                "restart_enabled": (enabled and not busy and remaining >= 3
                                    and (audit_locked or bool(campaign_rows))),
                "attempts_used": attempts_used, "remaining": remaining,
                "campaign_path": str(output_dir),
                "campaign_attempts": len(campaign_rows),
                "ledger_revision": [[row["id"], row["status"], row["created"]]
                                    for row in campaign_rows],
                "phase_counts": phase_counts,
                "operation": current}

    def restart_campaign():
        nonlocal config, ledger, output_dir, report_path, summary_path
        with operation_lock:
            if operation["state"] == "running":
                raise ValueError("Wait for the current training action to finish")
            previous_path = output_dir
            config, ledger = start_new_round(config, ledger)
            output_dir = config.output_dir
            report_path = output_dir / "report.html"
            summary_path = output_dir / "summary.json"
            operation.update(id=None, state="idle",
                             message=f"Started a new round; previous campaign is saved at {previous_path}.")
        return {"campaign_path": str(output_dir), "seed": config.seed,
                "attempts_used": ledger.used, "remaining": ledger.remaining}

    def start_operation(action, family=None):
        with operation_lock:
            if operation["state"] == "running":
                raise ValueError("A training action is already running")
            operation.update(id=secrets.token_hex(8), state="running",
                             message="Starting requested training action…")
            operation_id = operation["id"]

        def run():
            try:
                if config is None:
                    raise ValueError("Training actions are not configured for this server")
                config.permit.check()
                if action == "manual":
                    result = run_single_search_attempt(config, family)
                    summary = (f"Search attempt {result['id']} {result['status']} "
                               f"for {family}.")
                else:
                    result = run_campaign(config)
                    summary = (f"Full campaign finished with {result.get('attempts_used', ledger.used)} "
                               "attempt(s) recorded.")
                with operation_lock:
                    operation.update(state="finished", message=summary)
            except Exception as error:
                with operation_lock:
                    operation.update(state="failed", message=str(error),
                                     error_type=type(error).__name__)

        threading.Thread(target=run, name=f"autonomis-{action}-{operation_id}",
                         daemon=True).start()
        return operation_id

    class ReportHandler(BaseHTTPRequestHandler):
        def _json(self, status, value):
            content = json.dumps(value, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/api/status":
                try:
                    self._json(200, control_status())
                except Exception as error:
                    self._json(503, {"enabled": False, "reason": str(error)})
                return
            if path not in {"/", "/index.html"}:
                self.send_error(404, "Only the campaign report is available")
                return
            try:
                summary = read_json(summary_path) if summary_path.is_file() else None
                render_report(ledger, report_path, summary=summary, dataset_role=dataset_role,
                              training_controls=True, csrf_token=csrf_token)
                content = report_path.read_bytes()
            except (OSError, ValueError, RuntimeError) as error:
                content = f"Report unavailable: {error}".encode("utf-8")
                self.send_response(503)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def do_POST(self):
            path = urlsplit(self.path).path
            if path not in {"/api/train/manual", "/api/train/full", "/api/campaign/restart"}:
                self._json(404, {"error": "Unknown training action"})
                return
            origin = self.headers.get("Origin", "")
            expected_origin = f"http://{self.headers.get('Host', '')}"
            if origin != expected_origin or self.headers.get("X-Autonomis-Token") != csrf_token:
                self._json(403, {"error": "Request origin or page token is invalid"})
                return
            if config is None:
                self._json(403, {"error": "Training actions are not configured"})
                return
            try:
                if not is_loopback:
                    raise ValueError("Training controls require a loopback-bound server")
                config.permit.check()
                if ledger.identity.get("kind") != "real":
                    raise ValueError("Training controls are disabled for simulation ledgers")
                length = int(self.headers.get("Content-Length", "0"))
                if length < 0 or length > 4096:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError("Invalid training request")
                if path == "/api/campaign/restart":
                    result = restart_campaign()
                    self._json(200, {
                        "message": "New round started; the previous round is preserved in history.",
                        **result,
                    })
                    return
                if ledger.audit_lock is not None:
                    raise ValueError("This campaign's Audit is locked; start a new round to continue")
                if path.endswith("/manual"):
                    family = payload.get("family")
                    if family not in {"lr", "catboost", "lightgbm"}:
                        raise ValueError("Choose a supported model family")
                    action = "manual"
                else:
                    action, family = "full", None
                operation_id = start_operation(action, family)
                self._json(202, {"operation_id": operation_id,
                                 "message": "Training action accepted"})
            except (ValueError, RuntimeError) as error:
                self._json(409, {"error": str(error)})
            except Exception as error:
                self._json(403, {"error": str(error)})

        def log_message(self, format_string, *args):
            print(f"{self.address_string()} - {format_string % args}")

    server = ThreadingHTTPServer((host, port), ReportHandler)
    print(f"Autonomis local campaign demo: http://{host}:{server.server_port}")
    print("Refresh the page to read the latest saved campaign evidence. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    finally:
        server.server_close()
