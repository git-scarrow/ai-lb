"""Tests for task classification and routing.

Classification tests (deterministic):
  1. Simple code task → LOCAL_CODING_FAST_PATH
  2. Security term → SECURITY_SENSITIVE
  3. Repo-wide signal → STANDARD_CODING
  4. Code fence, short → LOCAL_CODING_FAST_PATH
  5. Multipart content (list) → classification works
  6. Empty messages → DEFAULT
  7. Long text without code signals → DEFAULT
  8. Code task + security term → SECURITY_SENSITIVE (security wins)
  9. Broad term alone → STANDARD_CODING
 10. Code task + broad term → STANDARD_CODING (broad wins over coding)

Routing tests:
 11. model_candidates_for_task returns correct candidates for LOCAL_CODING_FAST_PATH
 12. model_candidates_for_task returns empty list for DEFAULT (no config entry)
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "load_balancer" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from load_balancer.task_classification import TaskClass, classify_task
from load_balancer.task_routing import model_candidates_for_task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _payload(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}]}


def _multipart_payload(text: str) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                ],
            }
        ]
    }


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------

def test_simple_function_routes_local():
    payload = _payload("Write a Python function to parse CSV rows.")
    assert classify_task(payload) == TaskClass.LOCAL_CODING_FAST_PATH


def test_refactor_routes_local():
    payload = _payload("Refactor this function to use list comprehension.")
    assert classify_task(payload) == TaskClass.LOCAL_CODING_FAST_PATH


def test_crypto_routes_security_sensitive():
    payload = _payload("Review this JWT validation code for security bugs.")
    assert classify_task(payload) == TaskClass.SECURITY_SENSITIVE


def test_oauth_routes_security_sensitive():
    payload = _payload("Review this OAuth callback handler for account takeover risks.")
    assert classify_task(payload) == TaskClass.SECURITY_SENSITIVE


def test_repo_wide_routes_standard():
    payload = _payload("Find the root cause across this codebase.")
    assert classify_task(payload) == TaskClass.STANDARD_CODING


def test_architecture_routes_standard():
    payload = _payload("How should we architecture the new service?")
    assert classify_task(payload) == TaskClass.STANDARD_CODING


def test_code_fence_routes_local():
    payload = _payload("What does this do?\n```python\ndef foo(): pass\n```")
    assert classify_task(payload) == TaskClass.LOCAL_CODING_FAST_PATH


def test_empty_messages_routes_default():
    assert classify_task({"messages": []}) == TaskClass.DEFAULT


def test_missing_messages_routes_default():
    assert classify_task({}) == TaskClass.DEFAULT


def test_long_text_no_signals_routes_default():
    payload = _payload("a " * 7000)
    assert classify_task(payload) == TaskClass.DEFAULT


def test_security_overrides_coding():
    # Both coding and security terms present — security wins
    payload = _payload("Write a function to validate auth tokens using jwt.")
    assert classify_task(payload) == TaskClass.SECURITY_SENSITIVE


def test_broad_overrides_coding():
    # Both coding and broad terms present — STANDARD_CODING wins over LOCAL
    payload = _payload("Implement a function but we need to look across files.")
    assert classify_task(payload) == TaskClass.STANDARD_CODING


def test_multipart_content_routes_local():
    payload = _multipart_payload("Write a Python script to sort a list.")
    assert classify_task(payload) == TaskClass.LOCAL_CODING_FAST_PATH


def test_multipart_content_security():
    payload = _multipart_payload("Check this for sql injection vulnerabilities.")
    assert classify_task(payload) == TaskClass.SECURITY_SENSITIVE


# ---------------------------------------------------------------------------
# Routing tests
# ---------------------------------------------------------------------------

def test_model_candidates_for_local_coding():
    payload = _payload("Write a Python function to parse CSV rows.")
    task_class, candidates = model_candidates_for_task(payload)
    assert task_class == TaskClass.LOCAL_CODING_FAST_PATH
    assert "gemma4:26b-local" in candidates


def test_model_candidates_for_security():
    payload = _payload("Review this JWT validation for vulnerabilities.")
    task_class, candidates = model_candidates_for_task(payload)
    assert task_class == TaskClass.SECURITY_SENSITIVE
    assert len(candidates) > 0
    assert "gemma4:26b-local" not in candidates


def test_model_candidates_for_default_empty():
    # DEFAULT class has no config entry → empty candidates list
    with patch("load_balancer.task_routing.config") as mock_cfg:
        mock_cfg.MODEL_CLASSES = {}
        task_class, candidates = model_candidates_for_task({"messages": []})
        assert task_class == TaskClass.DEFAULT
        assert candidates == []


def test_model_candidates_missing_class_returns_empty():
    payload = _payload("Write a Python function.")
    with patch("load_balancer.task_routing.config") as mock_cfg:
        # Config has no local_coding_fast_path entry
        mock_cfg.MODEL_CLASSES = {}
        task_class, candidates = model_candidates_for_task(payload)
        assert task_class == TaskClass.LOCAL_CODING_FAST_PATH
        assert candidates == []
