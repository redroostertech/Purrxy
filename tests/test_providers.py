"""Tests for the provider registry, auth headers, URL validation and upstream selection."""

import unittest

from src.proxy.providers import (
    ANTHROPIC_VERSION,
    PRESETS,
    Upstream,
    UpstreamConfigError,
    build_auth_headers,
    endpoint_url,
    get_provider,
    is_loopback_host,
    redact,
    resolve_upstream,
    upstream_headers,
    validate_upstream_url,
)

KEY = "sk-unit-test-SECRET-0123456789"

EXPECTED_PRESETS = {
    # id: (dialect, auth, base_url or None)
    "openai": ("openai", "bearer", "https://api.openai.com/v1"),
    "anthropic": ("anthropic", "x-api-key", "https://api.anthropic.com/v1"),
    "gemini": ("openai", "bearer", "https://generativelanguage.googleapis.com/v1beta/openai"),
    "perplexity": ("openai", "bearer", "https://api.perplexity.ai"),
    "groq": ("openai", "bearer", "https://api.groq.com/openai/v1"),
    "together": ("openai", "bearer", "https://api.together.xyz/v1"),
    "mistral": ("openai", "bearer", "https://api.mistral.ai/v1"),
    "openrouter": ("openai", "bearer", "https://openrouter.ai/api/v1"),
    "deepseek": ("openai", "bearer", "https://api.deepseek.com/v1"),
    "xai": ("openai", "bearer", "https://api.x.ai/v1"),
    "azure_openai": ("openai", "api-key", None),
    "custom": ("openai", "bearer", None),
}


class TestRegistry(unittest.TestCase):
    def test_preset_set_is_exact(self):
        self.assertEqual(set(PRESETS), set(EXPECTED_PRESETS))

    def test_preset_fields(self):
        for pid, (dialect, auth, base_url) in EXPECTED_PRESETS.items():
            with self.subTest(provider=pid):
                p = PRESETS[pid]
                self.assertEqual(p.dialect, dialect)
                self.assertEqual(p.auth, auth)
                self.assertEqual(p.base_url, base_url)
                self.assertEqual(p.requires_url, base_url is None)
                if base_url:
                    self.assertTrue(base_url.startswith("https://"))
                    # every preset URL passes its own validation
                    self.assertEqual(validate_upstream_url(base_url, p), base_url)
                d = p.to_dict()
                self.assertEqual(d["id"], pid)
                self.assertIn("streaming", d["capabilities"])

    def test_anthropic_default_headers(self):
        self.assertEqual(
            PRESETS["anthropic"].default_headers, {"anthropic-version": ANTHROPIC_VERSION}
        )

    def test_get_provider_case_insensitive_and_unknown(self):
        self.assertIs(get_provider(" OpenAI "), PRESETS["openai"])
        with self.assertRaises(UpstreamConfigError):
            get_provider("nope")


class TestAuthHeaders(unittest.TestCase):
    def test_auth_style_per_provider(self):
        cases = [
            ("openai", {"Authorization": f"Bearer {KEY}"}),
            ("gemini", {"Authorization": f"Bearer {KEY}"}),
            ("anthropic", {"x-api-key": KEY}),
            ("azure_openai", {"api-key": KEY}),
        ]
        for pid, expected in cases:
            with self.subTest(provider=pid):
                self.assertEqual(build_auth_headers(PRESETS[pid], KEY), expected)

    def test_no_key_no_auth_header(self):
        self.assertEqual(build_auth_headers(PRESETS["custom"], None), {})

    def test_upstream_headers_full_set(self):
        cases = [
            (
                "anthropic",
                False,
                {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "anthropic-version": ANTHROPIC_VERSION,
                    "x-api-key": KEY,
                },
            ),
            (
                "openai",
                True,
                {
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                    "Authorization": f"Bearer {KEY}",
                },
            ),
        ]
        for pid, stream, expected in cases:
            with self.subTest(provider=pid):
                up = Upstream(PRESETS[pid], "https://example.com/v1", KEY)
                self.assertEqual(upstream_headers(up, stream=stream), expected)

    def test_repr_hides_key(self):
        up = Upstream(PRESETS["openai"], "https://api.openai.com/v1", KEY)
        self.assertNotIn(KEY, repr(up))
        self.assertNotIn(KEY, str(up))


class TestEndpointUrl(unittest.TestCase):
    def test_join(self):
        cases = [
            (
                "https://api.openai.com/v1",
                "/chat/completions",
                "https://api.openai.com/v1/chat/completions",
            ),
            (
                "https://api.openai.com/v1/",
                "chat/completions",
                "https://api.openai.com/v1/chat/completions",
            ),
            (
                "https://r.openai.azure.com/openai/deployments/gpt?api-version=2024-10-21",
                "/chat/completions",
                "https://r.openai.azure.com/openai/deployments/gpt/chat/completions?api-version=2024-10-21",
            ),
            (
                "https://api.perplexity.ai",
                "/chat/completions",
                "https://api.perplexity.ai/chat/completions",
            ),
        ]
        for base, suffix, expected in cases:
            with self.subTest(base=base):
                self.assertEqual(endpoint_url(base, suffix), expected)


class TestUrlValidation(unittest.TestCase):
    """Each rejected URL is paired with an accepted control case."""

    CASES = [
        # (reason, provider, rejected_url, accepted_control_url)
        ("http on public host", "openai", "http://api.openai.com/v1", "https://api.openai.com/v1"),
        (
            "http on private host even for custom",
            "custom",
            "http://10.0.0.5:8000/v1",
            "https://10.0.0.5:8000/v1",
        ),
        ("private IPv4 for non-custom", "openai", "https://10.1.2.3/v1", "https://8.8.8.8/v1"),
        ("192.168 for non-custom", "groq", "https://192.168.1.10/v1", "https://192.168.1.10/v1"),
        (
            "link-local metadata address",
            "anthropic",
            "https://169.254.169.254/latest",
            "https://127.0.0.1:9/v1",
        ),
        ("IPv6 unique local", "openai", "https://[fd00::1]/v1", "https://[::1]/v1"),
        ("IPv6 link-local", "openai", "https://[fe80::1]/v1", "https://[2001:4860:4860::8888]/v1"),
        (
            "IPv4-mapped private",
            "openai",
            "https://[::ffff:10.0.0.1]/v1",
            "https://[::ffff:8.8.8.8]/v1",
        ),
        (
            "legacy integer IPv4 for 10.0.0.1",
            "openai",
            "https://167772161/v1",
            "https://2130706433/v1",
        ),
        ("unspecified address", "openai", "https://0.0.0.0/v1", "http://localhost:8080/v1"),
        (
            "embedded credentials",
            "openai",
            "https://user:pw@api.openai.com/v1",
            "https://api.openai.com/v1",
        ),
        ("non-http scheme", "custom", "ftp://example.com/v1", "https://example.com/v1"),
        ("relative URL", "custom", "/v1/chat", "http://127.0.0.1:8080/v1"),
        ("fragment", "openai", "https://api.openai.com/v1#x", "https://api.openai.com/v1"),
    ]

    def test_rejected_and_controls(self):
        for reason, pid, bad, good in self.CASES:
            provider = PRESETS[pid]
            # the 192.168 case: rejected for groq, accepted for custom
            control_provider = PRESETS["custom"] if bad == good else provider
            with self.subTest(reason=reason, url=bad), self.assertRaises(UpstreamConfigError):
                validate_upstream_url(bad, provider)
            with self.subTest(reason=reason + " (control)", url=good):
                self.assertEqual(validate_upstream_url(good, control_provider), good.rstrip("/"))

    def test_loopback_detection(self):
        for host, expected in [
            ("127.0.0.1", True),
            ("localhost", True),
            ("::1", True),
            ("127.8.9.10", True),
            ("LOCALHOST", True),
            ("10.0.0.1", False),
            ("example.com", False),
            ("", False),
        ]:
            with self.subTest(host=host):
                self.assertEqual(is_loopback_host(host), expected)


class TestResolveUpstream(unittest.TestCase):
    def test_absent_headers_return_none(self):
        self.assertIsNone(resolve_upstream({"Content-Type": "application/json"}))

    def test_preset_default_url(self):
        up = resolve_upstream({"x-lana-upstream-provider": "groq", "X-Lana-Upstream-Key": KEY})
        self.assertEqual(up.provider.id, "groq")
        self.assertEqual(up.base_url, "https://api.groq.com/openai/v1")
        self.assertEqual(up.key, KEY)

    def test_url_override_and_custom_without_key(self):
        up = resolve_upstream(
            {
                "X-Lana-Upstream-Provider": "custom",
                "X-Lana-Upstream-Url": "http://127.0.0.1:8000/v1/",
            }
        )
        self.assertEqual(up.base_url, "http://127.0.0.1:8000/v1")
        self.assertIsNone(up.key)

    def test_invalid_combinations(self):
        cases = [
            ("key without provider", {"X-Lana-Upstream-Key": KEY}),
            ("url without provider", {"X-Lana-Upstream-Url": "https://x.com"}),
            ("unknown provider", {"X-Lana-Upstream-Provider": "zzz", "X-Lana-Upstream-Key": KEY}),
            ("custom requires url", {"X-Lana-Upstream-Provider": "custom"}),
            (
                "azure requires url",
                {"X-Lana-Upstream-Provider": "azure_openai", "X-Lana-Upstream-Key": KEY},
            ),
            ("preset requires key", {"X-Lana-Upstream-Provider": "openai"}),
            ("blank key", {"X-Lana-Upstream-Provider": "openai", "X-Lana-Upstream-Key": "  "}),
            (
                "bad url",
                {
                    "X-Lana-Upstream-Provider": "openai",
                    "X-Lana-Upstream-Key": KEY,
                    "X-Lana-Upstream-Url": "http://evil.example.com/v1",
                },
            ),
        ]
        for name, headers in cases:
            with self.subTest(case=name):
                with self.assertRaises(UpstreamConfigError) as ctx:
                    resolve_upstream(headers)
                self.assertNotIn(KEY, str(ctx.exception))


class TestRedact(unittest.TestCase):
    def test_redact(self):
        self.assertEqual(redact(f"bad key {KEY}!", KEY), "bad key [REDACTED]!")
        self.assertEqual(redact("nothing", None, ""), "nothing")
        self.assertEqual(redact("", KEY), "")


if __name__ == "__main__":
    unittest.main()
