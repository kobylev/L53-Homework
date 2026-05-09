"""Tests for Gatekeeper validation and sanitisation.

Adapted to the actual Gatekeeper API:

* ``validate_ticker(ticker) -> bool``       — returns True/False (does not raise).
* ``_sanitize_ticker(ticker) -> str``       — raises ValueError on invalid input.
* ``sanitize_filename(name) -> str``        — returns a safe basename;
                                              raises ValueError if no safe
                                              basename can be derived.
* Lowercase tickers are accepted via ``.upper()`` normalisation; this is a
  documented design choice and not a bug.

Run with:  pytest tests/test_gatekeeper.py -v
"""
from __future__ import annotations

import pytest

try:
    from src.gatekeeper import Gatekeeper
except Exception as exc:  # pragma: no cover
    Gatekeeper = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


# ----------------------- ticker whitelist (returns bool) ------------------


VALID_TICKERS = [
    "MSFT", "AAPL", "GOOG", "BRK.B", "BRK.A",
    "F", "T", "V", "GE",
    "msft",          # lowercase — accepted via .upper() normalisation
    "  AAPL  ",      # surrounding whitespace — accepted via .strip()
]

INVALID_TICKERS = [
    "",                       # empty
    "TOOLONG",                # > 5 letters
    "MSFT;DROP",              # SQL injection attempt
    "../etc/passwd",          # path traversal
    "MSFT/AAPL",              # separator
    "MSFT 1",                 # internal whitespace
    "M$FT",                   # special char
    "MSFT'",                  # quote
    "MSFT--",                 # SQL comment
    "<script>",               # XSS attempt
]


@pytest.mark.skipif(
    Gatekeeper is None, reason=f"Gatekeeper not importable: {_IMPORT_ERROR}"
)
@pytest.mark.parametrize("ticker", VALID_TICKERS)
def test_valid_tickers_accepted(ticker: str) -> None:
    """validate_ticker should return True for any well-formed ticker."""
    gk = Gatekeeper()
    assert gk.validate_ticker(ticker) is True, (
        f"Valid ticker rejected: {ticker!r}"
    )


@pytest.mark.skipif(
    Gatekeeper is None, reason=f"Gatekeeper not importable: {_IMPORT_ERROR}"
)
@pytest.mark.parametrize("ticker", INVALID_TICKERS)
def test_invalid_tickers_rejected(ticker: str) -> None:
    """validate_ticker should return False for any malformed input.

    The function MUST NOT silently accept SQL fragments, path traversal
    sequences, shell metacharacters, or oversized identifiers.
    """
    gk = Gatekeeper()
    assert gk.validate_ticker(ticker) is False, (
        f"validate_ticker accepted malicious or malformed input: {ticker!r}"
    )


@pytest.mark.skipif(
    Gatekeeper is None, reason=f"Gatekeeper not importable: {_IMPORT_ERROR}"
)
def test_validate_ticker_handles_non_string() -> None:
    """Non-string inputs must not raise; they must return False."""
    gk = Gatekeeper()
    assert gk.validate_ticker(None) is False
    assert gk.validate_ticker(123) is False
    assert gk.validate_ticker(["MSFT"]) is False


# ------------------- _sanitize_ticker (raises on invalid) -----------------


@pytest.mark.skipif(
    Gatekeeper is None, reason=f"Gatekeeper not importable: {_IMPORT_ERROR}"
)
def test_sanitize_ticker_raises_on_injection() -> None:
    """_sanitize_ticker is the strict variant that raises on invalid input.

    Used internally before constructing cache paths and API calls. It is
    the actual injection defense for downstream code.
    """
    gk = Gatekeeper()
    with pytest.raises(ValueError):
        gk._sanitize_ticker("MSFT; DROP TABLE users; --")
    with pytest.raises(ValueError):
        gk._sanitize_ticker("../etc/passwd")


@pytest.mark.skipif(
    Gatekeeper is None, reason=f"Gatekeeper not importable: {_IMPORT_ERROR}"
)
def test_sanitize_ticker_returns_uppercase() -> None:
    gk = Gatekeeper()
    assert gk._sanitize_ticker("msft") == "MSFT"
    assert gk._sanitize_ticker("  aapl  ") == "AAPL"


# ----------------------------- filename safety -----------------------------


PATH_TRAVERSAL_NAMES = [
    "../../etc/passwd",
    "..\\..\\windows\\system32\\config",
    "/absolute/path",
    "C:\\Windows\\malicious",
    "logs/../../secret.key",
    "normal/with/slash.txt",
]


@pytest.mark.skipif(
    Gatekeeper is None, reason=f"Gatekeeper not importable: {_IMPORT_ERROR}"
)
def test_sanitize_filename_method_exists() -> None:
    """sanitize_filename must be implemented on Gatekeeper."""
    gk = Gatekeeper()
    assert hasattr(gk, "sanitize_filename"), (
        "Gatekeeper.sanitize_filename is missing. "
        "See README's Gatekeeper section — apply the patch from CHANGELOG "
        "Round 2 to add the method to src/gatekeeper.py."
    )


@pytest.mark.skipif(
    Gatekeeper is None, reason=f"Gatekeeper not importable: {_IMPORT_ERROR}"
)
@pytest.mark.parametrize("name", PATH_TRAVERSAL_NAMES)
def test_filenames_stripped_of_path_components(name: str) -> None:
    """sanitize_filename must yield a string with no path components."""
    gk = Gatekeeper()
    if not hasattr(gk, "sanitize_filename"):
        pytest.skip("sanitize_filename not implemented yet — see patch.")
    safe = gk.sanitize_filename(name)
    assert "/" not in safe, f"sanitize_filename leaked '/': {safe!r}"
    assert "\\" not in safe, f"sanitize_filename leaked '\\\\': {safe!r}"
    assert ".." not in safe, f"sanitize_filename leaked '..': {safe!r}"
    assert ":" not in safe, f"sanitize_filename leaked ':': {safe!r}"


@pytest.mark.skipif(
    Gatekeeper is None, reason=f"Gatekeeper not importable: {_IMPORT_ERROR}"
)
def test_normal_filename_passes() -> None:
    gk = Gatekeeper()
    if not hasattr(gk, "sanitize_filename"):
        pytest.skip("sanitize_filename not implemented yet — see patch.")
    assert gk.sanitize_filename("MSFT_2024.csv") == "MSFT_2024.csv"


@pytest.mark.skipif(
    Gatekeeper is None, reason=f"Gatekeeper not importable: {_IMPORT_ERROR}"
)
def test_sanitize_filename_rejects_empty() -> None:
    gk = Gatekeeper()
    if not hasattr(gk, "sanitize_filename"):
        pytest.skip("sanitize_filename not implemented yet — see patch.")
    with pytest.raises(ValueError):
        gk.sanitize_filename("")
    with pytest.raises(ValueError):
        gk.sanitize_filename("   ")


# ------------------------------ rate limiting ------------------------------


@pytest.mark.skipif(
    Gatekeeper is None, reason=f"Gatekeeper not importable: {_IMPORT_ERROR}"
)
def test_rate_limit_attributes_exist() -> None:
    """Smoke check that the rate-limiter is wired up."""
    gk = Gatekeeper()
    assert hasattr(gk, "min_interval"), "Gatekeeper must expose min_interval"
    assert hasattr(gk, "last_call_time"), \
        "Gatekeeper must track last_call_time"
    assert gk.min_interval > 0, "min_interval must be positive"


# ------------ injection defense regression: hashing != validation ----------


@pytest.mark.skipif(
    Gatekeeper is None, reason=f"Gatekeeper not importable: {_IMPORT_ERROR}"
)
def test_validate_ticker_rejects_injection() -> None:
    """The actual injection defense is the regex, not the SHA-256 hash.

    SHA-256 of any string yields a valid hex string regardless of
    content, so hashing alone cannot block injection. The regex IS the
    defense; this test pins that behaviour.
    """
    gk = Gatekeeper()
    assert gk.validate_ticker("MSFT; DROP TABLE users; --") is False
    assert gk.validate_ticker("'; DELETE FROM trades --") is False
    assert gk.validate_ticker("$(rm -rf /)") is False
