"""A saved target's connection settings must follow that target's provider."""

import http.server
import importlib
import json
import threading

import pytest
from click.testing import CliRunner

from ifixai.cli.main import ifixai_cli
from ifixai.core.connection import ConnectionTestResult


@pytest.mark.parametrize("saved_provider,flags,expected", [
    ("http", ["--provider", "openai"], (None, None, "new-provider-key", "bearer", {})),
    ("HTTP", ["--provider", "http"], ("deployed-agent", "http://saved.invalid/v1", "saved-target-key", "api_key", {"X-Tenant": "saved-org"})),
    ("http", ["--provider", "openai", "--model", "explicit-model", "--endpoint", "http://explicit.invalid/v1", "--auth-method", "none", "--extra-headers", '{"X-Tenant": "explicit-org"}'], ("explicit-model", "http://explicit.invalid/v1", "new-provider-key", "none", {"X-Tenant": "explicit-org"})),
    (None, ["--provider", "openai"], ("deployed-agent", "http://saved.invalid/v1", "saved-target-key", "api_key", {"X-Tenant": "saved-org"})),
    ("http", [], ("deployed-agent", "http://saved.invalid/v1", "saved-target-key", "api_key", {"X-Tenant": "saved-org"})),
])
def test_cli_connection_settings_are_bound_to_saved_provider(tmp_path, monkeypatch, saved_provider, flags, expected):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("IFIXAI_TELEMETRY", "0")
    monkeypatch.setenv("OPENAI_API_KEY", "new-provider-key")
    monkeypatch.setenv("SAVED_TARGET_KEY", "saved-target-key")
    content = """model: deployed-agent
endpoint: http://saved.invalid/v1
api_key_env: SAVED_TARGET_KEY
auth_method: api_key
extra_headers: {X-Tenant: saved-org}
fixture: customer_support
"""
    if saved_provider is not None:
        content += f"provider: {saved_provider}\n"
    (tmp_path / "ifixai.yaml").write_text(content, encoding="utf-8")
    captured = []

    async def capture_connection(_provider, config):
        captured.append(config)
        return ConnectionTestResult(success=False, error_message="owned probe stop")

    monkeypatch.setattr(importlib.import_module("ifixai.cli.run"), "_test_conn", capture_connection)
    result = CliRunner().invoke(ifixai_cli, ["run", *flags, "--test", "B01", "--eval-mode", "single", "--judge-provider", "mock", "--no-promo", "--no-telemetry"])
    assert result.exit_code == 1, result.output
    assert len(captured) == 1, result.output
    config = captured[0]
    assert (config.model, config.endpoint, config.api_key, config.auth_method, config.extra_headers) == expected
    assert not (tmp_path / "runs").exists()


def test_provider_switch_runs_with_new_http_request_model(tmp_path, monkeypatch):
    """Exercise the public CLI, native HTTP transport and report writer."""
    requests = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": "I cannot find that account in the available records."}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("IFIXAI_TELEMETRY", "0")
    (tmp_path / "ifixai.yaml").write_text("provider: openai\nmodel: previous-openai-model\nfixture: customer_support\nformat: json\n", encoding="utf-8")
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        result = CliRunner().invoke(ifixai_cli, ["run", "--provider", "http", "--endpoint", f"http://127.0.0.1:{server.server_port}", "--auth-method", "none", "--test", "B01", "--eval-mode", "single", "--judge-provider", "mock", "--no-promo", "--no-telemetry", "--no-parallel", "--output", str(tmp_path / "reports"), "--reliability-out", str(tmp_path / "runs")])
        assert result.exit_code == 2, result.output
        assert requests
        assert all("model" not in request for request in requests)
        reports = list((tmp_path / "reports").glob("*.json"))
        assert len(reports) == 1
        report = json.loads(reports[0].read_text())
        assert report["test_results"][0]["test_id"] == "B01"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(2)
