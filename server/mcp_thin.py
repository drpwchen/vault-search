"""
vault-search — thin MCP client.
Author: P.W. Chen · https://drpwchen.com · https://github.com/drpwchen

A stdlib-only stdio MCP server that forwards every tool call to a running
api_server instead of opening the index itself.

Why: registering `mcp_server.py` directly gives every MCP client session its own
copy of the server, and each copy loads lancedb plus - on first use - the tokenizer
and the graph cache. Measured on the author's vault: 103 MB at import, 293 MB after
one vault_search, 338 MB with a textbook_search on top. (Before 2.7.0 made parent
lookup on-demand that last step reached 2.2 GB.) A few concurrent sessions still
pay hundreds of megabytes for one index. This client stays at 14 MB across the same
three steps and lets the one long-running api_server hold the index.

Use it instead of mcp_server.py when you run the API server anyway (e.g. for the
Obsidian plugin) or when you keep several agent sessions open at once:

    claude mcp add vault-search -- python "/abs/path/to/server/mcp_thin.py"

Tools, arguments and output are identical either way: the api_server dispatches
to the same handle_tool_call() function the standalone MCP server uses.

Configuration (all optional, same names config.py uses):
    VAULT_SEARCH_API_URL     full base URL of the API server
                             (default http://127.0.0.1:$VAULT_SEARCH_API_PORT)
    VAULT_SEARCH_API_PORT    port only, when the host is localhost (default 3789)
    VAULT_SEARCH_DATA_DIR    where api_key.txt and the tool-schema cache live
                             (default ~/.vault-search)
    VAULT_API_KEY            API key; falls back to <data dir>/api_key.txt

Failure behavior: if the API server is down, tools/list serves the last cached
schema so the session still starts, and tool calls return a plain-text error
telling the model the backend is unavailable rather than killing the session.
"""

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = os.environ.get("VAULT_SEARCH_API_URL") or (
    "http://127.0.0.1:" + os.environ.get("VAULT_SEARCH_API_PORT", "3789")
)
DATA_DIR = Path(os.environ.get("VAULT_SEARCH_DATA_DIR", "~/.vault-search")).expanduser()
KEY_FILE = DATA_DIR / "api_key.txt"
TOOLS_CACHE = DATA_DIR / "mcp_tools_cache.json"
CALL_TIMEOUT = 180  # embedding queries can queue behind other GPU work


def _api_key() -> str:
    key = os.environ.get("VAULT_API_KEY")
    if key:
        return key.strip()
    try:
        return KEY_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _request(path: str, payload: dict | None = None, timeout: int = CALL_TIMEOUT):
    req = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers={"Content-Type": "application/json", "X-API-Key": _api_key()},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_tools() -> list:
    try:
        tools = _request("/api/mcp/tools", timeout=15)["tools"]
        try:
            TOOLS_CACHE.parent.mkdir(parents=True, exist_ok=True)
            TOOLS_CACHE.write_text(json.dumps(tools, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
        return tools
    except Exception:
        # Serve the cached schema so the client session still starts.
        try:
            return json.loads(TOOLS_CACHE.read_text(encoding="utf-8"))
        except OSError:
            return []


def _call_tool(name: str, arguments: dict) -> str:
    try:
        return _request("/api/mcp/call", {"name": name, "arguments": arguments})["text"]
    except urllib.error.HTTPError as e:
        return json.dumps({"error": f"api_server returned HTTP {e.code}: {e.reason}"})
    except Exception as e:
        return json.dumps({
            "error": (
                f"vault-search api_server unreachable at {BASE_URL} "
                f"({type(e).__name__}: {e}). Start it with "
                "`python server/api_server.py`, or fall back to plain text search."
            )
        })


def _safe_dumps(obj) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(obj, ensure_ascii=False, default=str)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=False)
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")

    def send(obj):
        sys.stdout.write(_safe_dumps(obj) + "\n")
        sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = msg.get("method")
        msg_id = msg.get("id")

        if method == "initialize":
            send({
                "jsonrpc": "2.0", "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "vault-search", "version": "2.7.0-thin"},
                },
            })
        elif method == "notifications/initialized":
            pass
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": msg_id, "result": {"tools": _get_tools()}})
        elif method == "tools/call":
            name = msg["params"]["name"]
            arguments = msg["params"].get("arguments", {})
            send({
                "jsonrpc": "2.0", "id": msg_id,
                "result": {"content": [{"type": "text", "text": _call_tool(name, arguments)}]},
            })
        elif method == "ping":
            send({"jsonrpc": "2.0", "id": msg_id, "result": {}})
        else:
            if msg_id is not None:
                send({"jsonrpc": "2.0", "id": msg_id, "result": {}})


if __name__ == "__main__":
    main()
