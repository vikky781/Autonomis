"""Loopback HTTP server for the read-only campaign report."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from autonomis.reporting import render_report
from autonomis.storage import read_json


def serve_report(ledger, output_dir, *, dataset_role, host="127.0.0.1", port=8502):
    """Serve a freshly rendered report; requests never train or reserve attempts."""
    output_dir = Path(output_dir)
    report_path = output_dir / "report.html"
    summary_path = output_dir / "summary.json"

    class ReportHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if urlsplit(self.path).path not in {"/", "/index.html"}:
                self.send_error(404, "Only the campaign report is available")
                return
            try:
                summary = read_json(summary_path) if summary_path.is_file() else None
                render_report(ledger, report_path, summary=summary, dataset_role=dataset_role)
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

        def log_message(self, format_string, *args):
            print(f"{self.address_string()} - {format_string % args}")

    server = ThreadingHTTPServer((host, port), ReportHandler)
    print(f"Autonomis read-only demo: http://{host}:{server.server_port}")
    print("Refresh the page to read the latest saved campaign evidence. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    finally:
        server.server_close()
