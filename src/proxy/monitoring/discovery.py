"""
Auto-discovery of models available on backend servers.

Queries each backend's /v1/models endpoint at startup, logs discovered
models, and stores them in the Stats object for dashboard display.
"""

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, List, Set

from .stats import Stats

logger = logging.getLogger("proxy.discovery")


def discover_models(
    backend_urls: List[str],
    stats: Stats,
    timeout: int = 5,
) -> List[Dict[str, Any]]:
    """Query backend /v1/models endpoints and collect discovered models.

    Args:
        backend_urls: List of backend base URLs to query.
        stats: Stats object to store discovered models.
        timeout: HTTP timeout in seconds for discovery requests.

    Returns:
        List of model info dicts (each has at least an "id" field).
    """
    all_models: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()

    for base_url in backend_urls:
        url = f"{base_url.rstrip('/')}/v1/models"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read())

            models = body.get("data", [])
            for model in models:
                model_id = model.get("id", "unknown")
                if model_id not in seen_ids:
                    seen_ids.add(model_id)
                    all_models.append(model)
                    logger.info(
                        "Discovered model: %s (backend: %s)",
                        model_id, base_url,
                    )

        except urllib.error.URLError as e:
            logger.warning(
                "Model discovery failed for %s: %s", base_url, e
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(
                "Invalid model discovery response from %s: %s",
                base_url, e,
            )
        except Exception as e:
            logger.warning(
                "Unexpected error during model discovery for %s: %s",
                base_url, e,
            )

    if all_models:
        logger.info("Total discovered models: %d", len(all_models))
    else:
        logger.info("No models discovered from backends")

    stats.set_discovered_models(all_models)
    return all_models
