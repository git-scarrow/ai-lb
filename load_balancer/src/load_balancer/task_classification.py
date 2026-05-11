from enum import Enum
from typing import Any


class TaskClass(str, Enum):
    LOCAL_CODING_FAST_PATH = "local_coding_fast_path"
    STANDARD_CODING = "standard_coding"
    SECURITY_SENSITIVE = "security_sensitive"
    DEFAULT = "default"


CODING_TERMS = {
    "write", "implement", "fix", "debug", "refactor", "test", "tests",
    "function", "class", "method", "script", "typescript", "javascript",
    "python", "go", "rust", "java", "c++", "sql", "regex", "convert",
    "lint", "pytest", "unit test", "type error", "stack trace",
}

SECURITY_TERMS = {
    "auth", "oauth", "jwt", "saml", "crypto", "encryption", "signature",
    "iam", "policy", "sandbox", "deserialization", "pickle", "rce",
    "ssrf", "xss", "csrf", "sqli", "sql injection", "exploit",
    "payload", "malware", "phishing", "persistence", "evasion",
    "privilege escalation", "secret", "token leak", "incident",
}

BROAD_TERMS = {
    "architecture", "design the system", "repo", "codebase", "across files",
    "multiple files", "root cause", "production", "scalable", "migration",
    "large refactor", "end-to-end", "full implementation",
}


def _extract_text(payload: dict[str, Any]) -> str:
    messages = payload.get("messages") or []
    parts: list[str] = []

    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))

    return "\n".join(parts).lower()


def classify_task(payload: dict[str, Any]) -> TaskClass:
    text = _extract_text(payload)

    if not text.strip():
        return TaskClass.DEFAULT

    if any(term in text for term in SECURITY_TERMS):
        return TaskClass.SECURITY_SENSITIVE

    if any(term in text for term in BROAD_TERMS):
        return TaskClass.STANDARD_CODING

    looks_like_code_task = any(term in text for term in CODING_TERMS)
    has_code_fence = "```" in text
    short_enough = len(text) < 12_000

    if looks_like_code_task and short_enough:
        return TaskClass.LOCAL_CODING_FAST_PATH

    if has_code_fence and short_enough:
        return TaskClass.LOCAL_CODING_FAST_PATH

    return TaskClass.DEFAULT
