from __future__ import annotations

import re

EMAIL_MASK = "|||EMAIL_ADDRESS|||"
PHONE_MASK = "|||PHONE_NUMBER|||"
IP_MASK = "|||IP_ADDRESS|||"

EMAIL_RE = re.compile(
    r"""
    (?<![A-Za-z0-9._%+-])
    [A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+
    @
    (?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+
    [A-Za-z]{2,63}
    (?![A-Za-z0-9._%+-])
    """,
    re.VERBOSE,
)

PHONE_RE = re.compile(
    r"""
    (?<!\d)
    (?:\+?1[\s.-]?)?
    (?:
        \(\d{3}\)[\s.-]?
        |
        \d{3}[\s.-]?
    )
    \d{3}[\s.-]?\d{4}
    (?:\s*(?:ext\.?|x|\#)\s*\d{1,6})?
    (?!\d)
    """,
    re.VERBOSE | re.IGNORECASE,
)

IPV4_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
IPV4_RE = re.compile(
    rf"""
    (?<![\d.])
    {IPV4_OCTET}\.{IPV4_OCTET}\.{IPV4_OCTET}\.{IPV4_OCTET}
    (?!\.\d)
    (?!\d)
    """,
    re.VERBOSE,
)


def _mask_pattern(pattern: re.Pattern[str], text: str, replacement: str) -> tuple[str, int]:
    masked_text, num_masked = pattern.subn(replacement, text)
    return masked_text, num_masked


def mask_emails(text: str) -> tuple[str, int]:
    """Replace email addresses with a sentinel token and return the replacement count."""
    return _mask_pattern(EMAIL_RE, text, EMAIL_MASK)


def mask_phone_numbers(text: str) -> tuple[str, int]:
    """Replace common US phone number formats with a sentinel token."""
    return _mask_pattern(PHONE_RE, text, PHONE_MASK)


def mask_ips(text: str) -> tuple[str, int]:
    """Replace valid IPv4 addresses with a sentinel token."""
    return _mask_pattern(IPV4_RE, text, IP_MASK)
