"""
Provider registry and per-request upstream selection.

A provider preset describes how to reach one upstream API: its base URL, the
wire dialect it speaks ("openai" Chat Completions or "anthropic" Messages),
how the API key is presented, and which capabilities it supports.

Callers choose an upstream per request with these headers:

    X-Lana-Upstream-Provider   preset id (see PRESETS)
    X-Lana-Upstream-Url        optional base URL override (required for
                               "custom" and "azure_openai")
    X-Lana-Upstream-Key        provider API key

The key lives only on the Upstream object for the lifetime of one request.
It is excluded from repr(), never logged, never stored, and never part of a
cache or recorder key. X-Lana-* headers are never forwarded upstream; outgoing
headers are always built from scratch by upstream_headers().
"""

import ipaddress
import re
import socket
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional
from urllib.parse import urlsplit, urlunsplit

HEADER_PROVIDER = "X-Lana-Upstream-Provider"
HEADER_URL = "X-Lana-Upstream-Url"
HEADER_KEY = "X-Lana-Upstream-Key"

DIALECT_OPENAI = "openai"
DIALECT_ANTHROPIC = "anthropic"

AUTH_BEARER = "bearer"
AUTH_X_API_KEY = "x-api-key"
AUTH_API_KEY = "api-key"

ANTHROPIC_VERSION = "2023-06-01"


class UpstreamConfigError(ValueError):
    """Raised when per-request upstream headers are missing or invalid.

    Messages never contain the API key.
    """

    status = 400


@dataclass(frozen=True)
class Provider:
    """A provider preset."""

    id: str
    label: str
    base_url: Optional[str]
    dialect: str
    auth: str
    default_headers: Dict[str, str] = field(default_factory=dict)
    requires_url: bool = False
    requires_key: bool = True
    allow_private: bool = False
    tools: bool = True
    streaming: bool = True
    vision: bool = False
    models: bool = True
    notes: str = ""

    def to_dict(self) -> dict:
        """Public description of the preset (safe to expose; no secrets)."""
        return {
            "id": self.id,
            "label": self.label,
            "base_url": self.base_url,
            "dialect": self.dialect,
            "auth": self.auth,
            "requires_url": self.requires_url,
            "capabilities": {
                "tools": self.tools,
                "streaming": self.streaming,
                "vision": self.vision,
                "models": self.models,
            },
            "notes": self.notes,
        }


PRESETS: Dict[str, Provider] = {
    p.id: p
    for p in (
        Provider(
            id="openai",
            label="OpenAI",
            base_url="https://api.openai.com/v1",
            dialect=DIALECT_OPENAI,
            auth=AUTH_BEARER,
            vision=True,
        ),
        Provider(
            id="anthropic",
            label="Anthropic",
            base_url="https://api.anthropic.com/v1",
            dialect=DIALECT_ANTHROPIC,
            auth=AUTH_X_API_KEY,
            vision=True,
            default_headers={"anthropic-version": ANTHROPIC_VERSION},
            notes="Native Messages API; translated to and from Chat Completions.",
        ),
        Provider(
            id="gemini",
            label="Google Gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            dialect=DIALECT_OPENAI,
            auth=AUTH_BEARER,
            vision=True,
            notes="Google's OpenAI-compatible endpoint. A native generateContent "
            "adapter can come later.",
        ),
        Provider(
            id="perplexity",
            label="Perplexity",
            base_url="https://api.perplexity.ai",
            dialect=DIALECT_OPENAI,
            auth=AUTH_BEARER,
            tools=False,
            models=False,
            notes="Extra response fields (citations, search_results) pass through unchanged.",
        ),
        Provider(
            id="groq",
            label="Groq",
            base_url="https://api.groq.com/openai/v1",
            dialect=DIALECT_OPENAI,
            auth=AUTH_BEARER,
        ),
        Provider(
            id="together",
            label="Together AI",
            base_url="https://api.together.xyz/v1",
            dialect=DIALECT_OPENAI,
            auth=AUTH_BEARER,
        ),
        Provider(
            id="mistral",
            label="Mistral",
            base_url="https://api.mistral.ai/v1",
            dialect=DIALECT_OPENAI,
            auth=AUTH_BEARER,
        ),
        Provider(
            id="openrouter",
            label="OpenRouter",
            base_url="https://openrouter.ai/api/v1",
            dialect=DIALECT_OPENAI,
            auth=AUTH_BEARER,
            vision=True,
            notes="Capabilities vary by routed model.",
        ),
        Provider(
            id="deepseek",
            label="DeepSeek",
            base_url="https://api.deepseek.com/v1",
            dialect=DIALECT_OPENAI,
            auth=AUTH_BEARER,
        ),
        Provider(
            id="xai",
            label="xAI",
            base_url="https://api.x.ai/v1",
            dialect=DIALECT_OPENAI,
            auth=AUTH_BEARER,
            vision=True,
        ),
        Provider(
            id="azure_openai",
            label="Azure OpenAI",
            base_url=None,
            dialect=DIALECT_OPENAI,
            auth=AUTH_API_KEY,
            requires_url=True,
            vision=True,
            models=False,
            notes="X-Lana-Upstream-Url is required, e.g. https://RESOURCE.openai.azure.com"
            "/openai/deployments/DEPLOYMENT?api-version=2024-10-21 (the query "
            "string is preserved).",
        ),
        Provider(
            id="custom",
            label="Custom (OpenAI-compatible)",
            base_url=None,
            dialect=DIALECT_OPENAI,
            auth=AUTH_BEARER,
            requires_url=True,
            requires_key=False,
            allow_private=True,
            notes="Any OpenAI-compatible server (vLLM, llama.cpp, Fireworks, ...). "
            "Private addresses are allowed; https is still required except "
            "on loopback.",
        ),
    )
}

# Config-routed backend (the pre-gateway behavior). Not selectable by header.
LOCAL_BACKEND = Provider(
    id="backend",
    label="Configured backend",
    base_url=None,
    dialect=DIALECT_OPENAI,
    auth=AUTH_BEARER,
    requires_key=False,
    allow_private=True,
)


@dataclass
class Upstream:
    """A resolved upstream for one request. The key is never part of repr()."""

    provider: Provider
    base_url: str
    key: Optional[str] = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return self.provider.id

    @property
    def host(self) -> str:
        return urlsplit(self.base_url).hostname or ""


def get_provider(provider_id: str) -> Provider:
    """Look up a preset by id (case-insensitive)."""
    provider = PRESETS.get((provider_id or "").strip().lower())
    if provider is None:
        raise UpstreamConfigError(
            f"Unknown upstream provider '{provider_id}'. "
            f"Known providers: {', '.join(sorted(PRESETS))}"
        )
    return provider


def build_auth_headers(provider: Provider, key: Optional[str]) -> Dict[str, str]:
    """Return the auth header for the provider's auth style (empty when no key)."""
    if not key:
        return {}
    if provider.auth == AUTH_BEARER:
        return {"Authorization": f"Bearer {key}"}
    if provider.auth == AUTH_X_API_KEY:
        return {"x-api-key": key}
    if provider.auth == AUTH_API_KEY:
        return {"api-key": key}
    raise ValueError(f"Unknown auth style: {provider.auth}")


def upstream_headers(upstream: Upstream, stream: bool = False) -> Dict[str, str]:
    """Build the complete outgoing header set. Client headers are never copied."""
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json",
    }
    headers.update(upstream.provider.default_headers)
    headers.update(build_auth_headers(upstream.provider, upstream.key))
    return headers


def endpoint_url(base_url: str, suffix: str) -> str:
    """Append an endpoint path to a base URL, preserving any query string.

    endpoint_url("https://x/v1", "/chat/completions") -> "https://x/v1/chat/completions"
    endpoint_url("https://x/d/m?api-version=1", "/chat/completions")
        -> "https://x/d/m/chat/completions?api-version=1"
    """
    parts = urlsplit(base_url)
    path = parts.path.rstrip("/") + "/" + suffix.lstrip("/")
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))


_LOOPBACK_NAMES = {"localhost"}
_LEGACY_IPV4 = re.compile(r"^(0x[0-9a-f]+|[0-9]+)(\.(0x[0-9a-f]+|[0-9]+)){0,3}$", re.I)


def _parse_ip(host: str):
    """Parse an IP literal, including legacy IPv4 forms like 2130706433."""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        if not _LEGACY_IPV4.match(host):
            return None
        try:
            ip = ipaddress.IPv4Address(socket.inet_aton(host))
        except OSError:
            return None
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip


def is_loopback_host(host: str) -> bool:
    """True for localhost and loopback IP literals (127.0.0.0/8, ::1)."""
    host = (host or "").lower().rstrip(".")
    if host in _LOOPBACK_NAMES:
        return True
    ip = _parse_ip(host)
    return ip is not None and ip.is_loopback


def validate_upstream_url(url: str, provider: Provider) -> str:
    """Validate an upstream base URL and return it without a trailing slash.

    Rules:
      - scheme must be https, except http is allowed for loopback hosts
        (127.0.0.1, localhost, ::1)
      - no embedded credentials (user:pass@host)
      - for presets other than "custom": private, link-local, unspecified,
        multicast and reserved IP literals are refused (loopback is allowed)

    Hostnames are not DNS-resolved; only IP literals are classified.
    """
    if not url or not isinstance(url, str):
        raise UpstreamConfigError("Upstream URL is empty")
    try:
        parts = urlsplit(url.strip())
        host = parts.hostname or ""
        parts.port  # noqa: B018 (raises ValueError on an invalid port)
    except ValueError as e:
        raise UpstreamConfigError(f"Upstream URL is malformed: {e}") from None
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https") or not host:
        raise UpstreamConfigError("Upstream URL must be an absolute http(s) URL")
    if parts.username is not None or parts.password is not None:
        raise UpstreamConfigError("Upstream URL must not contain credentials")
    if parts.fragment:
        raise UpstreamConfigError("Upstream URL must not contain a fragment")

    loopback = is_loopback_host(host)
    if scheme != "https" and not loopback:
        raise UpstreamConfigError(
            f"Upstream URL must use https (http is only allowed for loopback), got host {host}"
        )

    if not loopback and not provider.allow_private:
        ip = _parse_ip(host)
        if ip is not None and (
            ip.is_private
            or ip.is_link_local
            or ip.is_unspecified
            or ip.is_multicast
            or ip.is_reserved
        ):
            raise UpstreamConfigError(
                f"Upstream URL host {host} is a private or link-local address; "
                f"only the 'custom' provider may target private networks"
            )
    return url.strip().rstrip("/")


def _header(headers: Mapping[str, str], name: str) -> Optional[str]:
    """Case-insensitive header lookup that works for dicts and HTTPMessage."""
    getter = getattr(headers, "get", None)
    value = getter(name) if getter else None
    if value is None and isinstance(headers, dict):
        lowered = name.lower()
        for k, v in headers.items():
            if k.lower() == lowered:
                value = v
                break
    if value is None:
        return None
    value = value.strip()
    return value or None


def resolve_upstream(headers: Mapping[str, str]) -> Optional[Upstream]:
    """Resolve the per-request upstream from X-Lana-Upstream-* headers.

    Returns None when no X-Lana-Upstream-* header is present, meaning the
    existing config-based backend routing applies.

    Raises:
        UpstreamConfigError: if the headers are inconsistent or invalid.
    """
    provider_id = _header(headers, HEADER_PROVIDER)
    url = _header(headers, HEADER_URL)
    key = _header(headers, HEADER_KEY)

    if provider_id is None:
        if url is not None or key is not None:
            raise UpstreamConfigError(
                f"{HEADER_PROVIDER} is required when other X-Lana-Upstream-* headers are sent"
            )
        return None

    provider = get_provider(provider_id)
    if url is None:
        if provider.requires_url or not provider.base_url:
            raise UpstreamConfigError(f"{HEADER_URL} is required for provider '{provider.id}'")
        url = provider.base_url
    base_url = validate_upstream_url(url, provider)

    if key is None and provider.requires_key:
        raise UpstreamConfigError(f"{HEADER_KEY} is required for provider '{provider.id}'")

    return Upstream(provider=provider, base_url=base_url, key=key)


def redact(text: str, *secrets: Optional[str]) -> str:
    """Replace every occurrence of each secret in text with [REDACTED]."""
    if not text:
        return text
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text
