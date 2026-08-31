from __future__ import annotations

import unittest

from src.on_device_authority import (
    AUTHORITY_EVIDENCE_STATE,
    AuthorityRequest,
    evaluate_authority,
)


class OnDeviceAuthorityTests(unittest.TestCase):
    def test_local_operation_is_allowed_without_network_authority(self) -> None:
        result = evaluate_authority(AuthorityRequest("plan_kv", "personal"))
        self.assertEqual(result["decision"], "ALLOW_LOCAL")
        self.assertEqual(result["evidence_state"], AUTHORITY_EVIDENCE_STATE)
        self.assertFalse(result["network_execution"])
        self.assertFalse(result["ane_execution"])
        self.assertFalse(result["operational_authority"])
        self.assertEqual(len(result["receipt_sha256"]), 64)

    def test_sensitive_network_egress_is_denied_even_if_confirmed(self) -> None:
        result = evaluate_authority(
            AuthorityRequest(
                "export_receipt",
                "sensitive",
                network_required=True,
                user_confirmed_network=True,
            )
        )
        self.assertEqual(result["decision"], "DENY_NETWORK")

    def test_non_sensitive_network_requires_explicit_user_confirmation(self) -> None:
        pending = evaluate_authority(
            AuthorityRequest("export_receipt", "personal", network_required=True)
        )
        confirmed = evaluate_authority(
            AuthorityRequest(
                "export_receipt",
                "personal",
                network_required=True,
                user_confirmed_network=True,
            )
        )
        self.assertEqual(pending["decision"], "CONFIRM_NETWORK")
        self.assertTrue(pending["requires_user_confirmation"])
        self.assertEqual(confirmed["decision"], "ALLOW_CONFIRMED_NETWORK")
        self.assertFalse(confirmed["requires_user_confirmation"])

    def test_unsupported_operation_and_bad_flags_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_authority(AuthorityRequest("delete_everything", "public"))
        with self.assertRaises(ValueError):
            evaluate_authority(
                AuthorityRequest(
                    "plan_kv",
                    "public",
                    network_required="yes",  # type: ignore[arg-type]
                )
            )


if __name__ == "__main__":
    unittest.main()
