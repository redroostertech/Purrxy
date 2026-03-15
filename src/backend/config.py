"""
llama.cpp backend configuration and discovery.

This module is the Purrxy-specific glue between the generic supervisor
and the llama.cpp backend. It handles:

    1. Finding the llama-server binary across known build locations.
    2. Building the command-line arguments from .env configuration.
    3. Validating that the binary and model file exist before launch.

Architecture
============

    This is the ONLY module that knows about llama.cpp specifics. The
    supervisor module is generic and reusable — it just runs commands.
    This module translates purrxy's .env config into a concrete
    command that the supervisor can execute.

    .env (via proxy/env.py)  -->  BackendConfig  -->  build_backend_command()  -->  List[str] argv
                                                                                        |
                                                                      supervisor.ProcessConfig(cmd=argv)

Binary Search Order
===================

    find_llama_server() checks these paths relative to LLAMA_CPP_DIR:

        build/bin/llama-server    (standard cmake build)
        build/bin/server          (older llama.cpp versions)
        build/llama-server        (some build configs)
        llama-server              (manual placement / symlink)

    First executable match wins. Returns None if nothing found.

Environment Variables
=====================

    All read from os.environ (populate with load_dotenv_into_environ first):

    LLAMA_CPP_DIR     — Root of llama.cpp clone (default: ~/llama.cpp)
    MODEL_DIR         — Directory containing .gguf files (default: ~/llama.cpp/models)
    MODEL_NAME        — Filename of the model to load
    CONTEXT_SIZE      — Token context window (default: 8192)
    PARALLEL_SLOTS    — Concurrent inference slots (default: 1)
    GPU_LAYERS        — Layers to offload to GPU (default: 99 = all)
    FLASH_ATTENTION   — Enable flash attention (default: true)
    BACKEND_HOST      — Bind address (default: 127.0.0.1)
    BACKEND_PORT      — Listen port (default: 8080)
"""

import os
from dataclasses import dataclass
from typing import List, Optional


# Candidate paths for the llama-server binary, relative to LLAMA_CPP_DIR
LLAMA_SERVER_CANDIDATES = [
    "build/bin/llama-server",
    "build/bin/server",
    "build/llama-server",
    "llama-server",
]


@dataclass
class BackendConfig:
    """Configuration for the llama.cpp backend process."""

    llama_cpp_dir: str
    model_dir: str
    model_name: str
    context_size: int = 8192
    parallel_slots: int = 1
    gpu_layers: int = 99
    flash_attention: bool = True
    host: str = "127.0.0.1"
    port: int = 8080

    @classmethod
    def from_env(cls) -> "BackendConfig":
        """Build configuration from environment variables.

        Call load_dotenv_into_environ() before this to populate os.environ
        from .env file.
        """
        home = os.path.expanduser("~")

        return cls(
            llama_cpp_dir=os.path.expanduser(
                os.environ.get("LLAMA_CPP_DIR", os.path.join(home, "llama.cpp"))
            ),
            model_dir=os.path.expanduser(
                os.environ.get("MODEL_DIR", os.path.join(home, "llama.cpp", "models"))
            ),
            model_name=os.environ.get(
                "MODEL_NAME", "Qwen2.5-Coder-14B-Instruct-Q5_K_M.gguf"
            ),
            context_size=int(os.environ.get("CONTEXT_SIZE", "8192")),
            parallel_slots=int(os.environ.get("PARALLEL_SLOTS", "1")),
            gpu_layers=int(os.environ.get("GPU_LAYERS", "99")),
            flash_attention=os.environ.get("FLASH_ATTENTION", "true").lower()
            == "true",
            host=os.environ.get("BACKEND_HOST", "127.0.0.1"),
            port=int(os.environ.get("BACKEND_PORT", "8080")),
        )

    @property
    def model_path(self) -> str:
        """Full path to the model file."""
        return os.path.join(self.model_dir, self.model_name)

    @property
    def health_url(self) -> str:
        """Health check URL for the backend."""
        return f"http://{self.host}:{self.port}/health"

    def validate(self) -> None:
        """Validate that required files exist.

        Raises:
            FileNotFoundError: If llama-server binary or model file not found.
        """
        server = find_llama_server(self.llama_cpp_dir)
        if not server:
            raise FileNotFoundError(
                f"llama-server not found in {self.llama_cpp_dir}. "
                "Run install.sh first."
            )

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"Model not found: {self.model_path}. "
                "Run install.sh to download it."
            )


def find_llama_server(llama_cpp_dir: str) -> Optional[str]:
    """Search known locations for the llama-server binary.

    Args:
        llama_cpp_dir: Root directory of the llama.cpp installation.

    Returns:
        Absolute path to the binary, or None if not found.
    """
    llama_cpp_dir = os.path.expanduser(llama_cpp_dir)

    for candidate in LLAMA_SERVER_CANDIDATES:
        path = os.path.join(llama_cpp_dir, candidate)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return os.path.abspath(path)

    return None


def build_backend_command(config: BackendConfig) -> List[str]:
    """Build the command line to launch the llama.cpp server.

    Args:
        config: Backend configuration.

    Returns:
        List of command arguments for subprocess.

    Raises:
        FileNotFoundError: If llama-server binary not found.
    """
    server = find_llama_server(config.llama_cpp_dir)
    if not server:
        raise FileNotFoundError(
            f"llama-server not found in {config.llama_cpp_dir}"
        )

    cmd = [
        server,
        "--model", config.model_path,
        "--host", config.host,
        "--port", str(config.port),
        "--ctx-size", str(config.context_size),
        "--n-gpu-layers", str(config.gpu_layers),
        "--parallel", str(config.parallel_slots),
    ]

    if config.flash_attention:
        cmd.extend(["--flash-attn", "on"])

    return cmd
