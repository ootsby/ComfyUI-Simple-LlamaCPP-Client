from .simple_llamacpp_client import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .server_routes import setup_routes

setup_routes()

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
