#!/usr/bin/env python3
"""Composite MCP server that hosts multiple MCP services on one HTTP server.

Each service lives at its own path (e.g. /llm-api-search/mcp) so users can
connect to only the ones they need.

Run:  python mcp_server.py --http --port 8080

Browse available servers:  curl http://YOUR_VPS:8080/
Connect in Claude Code:    claude mcp add --transport url --scope user llm-api-search http://YOUR_VPS:8080/llm-api-search/mcp
"""

import argparse
import contextlib
import importlib
import pkgutil
from contextlib import asynccontextmanager

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Mount, Route


def discover_servers():
    """Auto-discover all MCP server modules in the mcp_servers package."""
    import mcp_servers

    servers = {}
    for importer, modname, ispkg in pkgutil.iter_modules(mcp_servers.__path__):
        if modname.startswith("_"):
            continue
        mod = importlib.import_module(f"mcp_servers.{modname}")
        if hasattr(mod, "mcp") and hasattr(mod, "MOUNT_PATH"):
            servers[modname] = {
                "module": mod,
                "mcp": mod.mcp,
                "mount_path": mod.MOUNT_PATH,
                "description": getattr(mod, "DESCRIPTION", ""),
            }
    return servers


def check_health(servers: dict) -> tuple[bool, dict]:
    """Determine whether the service is healthy.

    Verifies MCP servers are registered and every provider's static catalog
    is parseable. No network calls — deploy health stays decoupled from
    upstream provider API availability.
    """
    from llm_api_search.providers import PROVIDERS

    if not servers:
        return False, {"reason": "no MCP servers registered"}

    provider_status: dict[str, dict] = {}
    for name, cls in PROVIDERS.items():
        try:
            info = cls().get_static_info()
        except Exception as e:
            return False, {"reason": f"provider {name} failed get_static_info: {e}"}
        if not info.models:
            return False, {"reason": f"provider {name} has no models"}
        provider_status[name] = {"models": len(info.models)}

    return True, {
        "servers": list(servers.keys()),
        "providers": provider_status,
    }


def build_app(host: str, port: int):
    """Build a Starlette app that mounts all discovered MCP servers."""
    servers = discover_servers()

    mounts = []
    registry = {}
    session_managers = []

    for name, info in servers.items():
        mcp_instance = info["mcp"]
        mount_path = info["mount_path"]

        # Each MCP server's sub-app gets mounted at its own path
        sub_app = mcp_instance.streamable_http_app()
        mounts.append(Mount(mount_path, app=sub_app))
        session_managers.append(mcp_instance.session_manager)

        registry[name] = {
            "path": mount_path,
            "mcp_endpoint": f"{mount_path}/mcp",
            "description": info["description"],
            "claude_code_command": (
                f"claude mcp add --transport url --scope user {name} "
                f"http://YOUR_VPS:{port}{mount_path}/mcp"
            ),
        }

    async def index(request: Request):
        """List all available MCP servers and how to connect."""
        accept = request.headers.get("accept", "")
        if "application/json" in accept:
            return JSONResponse({"servers": registry})

        lines = [
            "Available MCP Servers",
            "=" * 50,
            "",
        ]
        for name, info in registry.items():
            lines.append(f"  {name}")
            lines.append(f"    {info['description']}")
            lines.append(f"    Endpoint: {info['mcp_endpoint']}")
            lines.append(f"    Connect:  {info['claude_code_command']}")
            lines.append("")

        return PlainTextResponse("\n".join(lines))

    async def health(request: Request):
        ok, details = check_health(servers)
        return JSONResponse(
            {"status": "ok" if ok else "unhealthy", **details},
            status_code=200 if ok else 503,
        )

    all_routes: list[Route | Mount] = [
        Route("/", index),
        Route("/health", health),
    ] + mounts

    @asynccontextmanager
    async def lifespan(app):
        async with contextlib.AsyncExitStack() as stack:
            for sm in session_managers:
                await stack.enter_async_context(sm.run())
            yield

    return Starlette(routes=all_routes, lifespan=lifespan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM API Search MCP Server")
    parser.add_argument(
        "--http", action="store_true", help="Run as HTTP server (streamable-http)"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    args = parser.parse_args()

    if args.http:
        import uvicorn

        app = build_app(args.host, args.port)
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        # Stdio mode — run the first discovered server (single-server mode)
        servers = discover_servers()
        if servers:
            first = next(iter(servers.values()))
            first["mcp"].run()
        else:
            print("No MCP servers found in mcp_servers/")
