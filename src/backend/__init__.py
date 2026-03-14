"""
llama.cpp backend configuration and discovery.

Handles finding the llama-server binary, building command-line arguments,
and validating the model file. This is the only module that knows about
llama.cpp specifics.

Classes:
    BackendConfig   Configuration dataclass for the llama.cpp server.

Functions:
    find_llama_server       Search known paths for the binary.
    build_backend_command   Build the subprocess argv from config.
"""

from .config import BackendConfig, find_llama_server, build_backend_command

__all__ = [
    "BackendConfig",
    "find_llama_server",
    "build_backend_command",
]
