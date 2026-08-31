"""Local authority capsule for Apple-Silicon-oriented portfolio workflows.

The capsule evaluates whether a modeled operation remains local, requires
explicit network confirmation, or is denied. It performs no network or hardware
operation and does not represent Apple platform authority.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum

AUTHORITY_SCHEMA = "glaciereq.apple-local-authority.v1"
AUTHORITY_EVIDENCE_STATE = "LOCAL_PRIVACY_AUTHORITY_POLICY_NOT_APPLE_PLATFORM_AUTHORITY"


class DataClass(StrEnum):
    PUBLIC = "public"
    PERSONAL = "personal"
    SENSITIVE = "sensitive"


class AuthorityDecision(StrEnum):
    ALLOW_LOCAL = "ALLOW_LOCAL"
    ALLOW_CONFIRMED_NETWORK = "ALLOW_CONFIRMED_NETWORK"
    CONFIRM_NETWORK = "CONFIRM_NETWORK"
    DENY_NETWORK = "DENY_NETWORK"


_ALLOWED_OPERATIONS = frozenset({"plan_kv", "quantize_local", "export_receipt"})


@dataclass(frozen=True, slots=True)
class AuthorityRequest:
    operation: str
    data_class: DataClass | str
    network_required: bool = False
    user_confirmed_network: bool = False

    def normalized(self) -> tuple[str, DataClass]:
        if not isinstance(self.operation, str) or not self.operation.strip():
            raise ValueError("operation must be non-empty text")
        operation = self.operation.strip()
        if operation not in _ALLOWED_OPERATIONS:
            raise ValueError(f"unsupported operation: {operation}")
        try:
            data_class = (
                self.data_class
                if isinstance(self.data_class, DataClass)
                else DataClass(self.data_class)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported data_class") from exc
        if not isinstance(self.network_required, bool):
            raise ValueError("network_required must be boolean")
        if not isinstance(self.user_confirmed_network, bool):
            raise ValueError("user_confirmed_network must be boolean")
        return operation, data_class


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def evaluate_authority(request: AuthorityRequest) -> dict[str, object]:
    """Return a deterministic privacy/authority decision without executing work."""

    operation, data_class = request.normalized()
    if not request.network_required:
        decision = AuthorityDecision.ALLOW_LOCAL
        reason = "operation remains within the declared local boundary"
    elif data_class is DataClass.SENSITIVE:
        decision = AuthorityDecision.DENY_NETWORK
        reason = "sensitive data is denied network egress by this local policy"
    elif request.user_confirmed_network:
        decision = AuthorityDecision.ALLOW_CONFIRMED_NETWORK
        reason = "non-sensitive network egress has explicit user confirmation"
    else:
        decision = AuthorityDecision.CONFIRM_NETWORK
        reason = "network egress requires explicit user confirmation"

    body: dict[str, object] = {
        "schema": AUTHORITY_SCHEMA,
        "operation": operation,
        "data_class": data_class.value,
        "network_required": request.network_required,
        "user_confirmed_network": request.user_confirmed_network,
        "decision": decision.value,
        "reason": reason,
        "requires_user_confirmation": decision is AuthorityDecision.CONFIRM_NETWORK,
        "evidence_state": AUTHORITY_EVIDENCE_STATE,
        "operational_authority": False,
        "network_execution": False,
        "ane_execution": False,
    }
    body["receipt_sha256"] = _digest(body)
    return body
