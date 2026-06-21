import requests
from server import PromptServer
from aiohttp import web


def setup_routes():
    @PromptServer.instance.routes.get("/llamacpp/models")
    async def get_models(request):
        url = request.query.get("url", "http://127.0.0.1:8888").rstrip("/")
        try:
            r = requests.get(f"{url}/models", timeout=5)
            r.raise_for_status()
            data = r.json().get("data", [])
            models = []
            for m in data:
                entry = {
                    "id": m.get("id", ""),
                    "status": (m.get("status") or {}).get("value", "unknown"),
                    "input_modalities": (m.get("architecture") or {}).get("input_modalities", []),
                    "output_modalities": (m.get("architecture") or {}).get("output_modalities", []),
                }
                models.append(entry)
            return web.json_response({"models": models, "status": "ok"})
        except Exception as e:
            return web.json_response({"models": [], "status": "error", "error": str(e)})

    @PromptServer.instance.routes.get("/llamacpp/running")
    async def get_running(request):
        url = request.query.get("url", "http://127.0.0.1:8888").rstrip("/")
        try:
            r = requests.get(f"{url}/models", timeout=5)
            r.raise_for_status()
            data = r.json().get("data", [])
            running = [
                m.get("id", "")
                for m in data
                if (m.get("status") or {}).get("value") == "loaded"
            ]
            return web.json_response({"running": running})
        except Exception as e:
            return web.json_response({"running": [], "error": str(e)})

    @PromptServer.instance.routes.get("/llamacpp/unload")
    async def do_unload(request):
        url = request.query.get("url", "http://127.0.0.1:8888").rstrip("/")
        model = request.query.get("model", "")
        try:
            r = requests.post(f"{url}/models/unload", json={"model": model}, timeout=10)
            return web.json_response({"status": "ok", "message": r.text.strip()})
        except Exception as e:
            return web.json_response({"status": "error", "error": str(e)})

    @PromptServer.instance.routes.get("/llamacpp/load")
    async def do_load(request):
        url = request.query.get("url", "http://127.0.0.1:8888").rstrip("/")
        model = request.query.get("model", "")
        try:
            r = requests.post(f"{url}/models/load", json={"model": model}, timeout=10)
            return web.json_response({"status": "ok", "message": r.text.strip()})
        except Exception as e:
            return web.json_response({"status": "error", "error": str(e)})
