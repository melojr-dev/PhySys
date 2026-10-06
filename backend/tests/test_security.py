import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.security import AuthError, AuthManager, AuditStore


class AuthManagerTests(unittest.TestCase):
    def setUp(self):
        self.auth = AuthManager(
            secret="a-very-long-authentication-secret-123456",
            username="clinician",
            password="correct-horse-battery-staple",
            role="admin",
            ttl_seconds=3600,
            pseudonym_secret="another-long-pseudonym-secret-654321",
        )

    def test_login_and_token_roundtrip(self):
        user = self.auth.authenticate("clinician", "correct-horse-battery-staple")
        token, expires = self.auth.issue_token(user)
        self.assertGreater(expires, 0)
        decoded = self.auth.verify_token(token)
        self.assertEqual(decoded.username, "clinician")
        self.assertEqual(decoded.role, "admin")

    def test_invalid_password(self):
        with self.assertRaises(AuthError):
            self.auth.authenticate("clinician", "wrong")

    def test_tampered_token_is_rejected(self):
        user = self.auth.authenticate("clinician", "correct-horse-battery-staple")
        token, _ = self.auth.issue_token(user)
        with self.assertRaises(AuthError):
            self.auth.verify_token(token[:-1] + ("A" if token[-1] != "A" else "B"))

    def test_pseudonym_is_deterministic_and_non_plaintext(self):
        a = self.auth.pseudonymize_patient_id("Paciente 001")
        b = self.auth.pseudonymize_patient_id(" paciente   001 ")
        self.assertEqual(a, b)
        self.assertTrue(a.startswith("P-"))
        self.assertNotIn("PACIENTE", a)

    def test_pseudonymization_is_idempotent(self):
        a = self.auth.pseudonymize_patient_id("Paciente 001")
        self.assertEqual(self.auth.pseudonymize_patient_id(a), a)

    def test_production_rejects_example_secrets(self):
        with patch.dict(
            "os.environ",
            {
                "PHYSIOAI_ENV": "production",
                "PHYSIOAI_AUTH_SECRET": "troque-por-ao-menos-32-caracteres-aleatorios",
                "PHYSIOAI_ADMIN_PASSWORD": "troque-por-uma-senha-longa-e-unica",
                "PHYSIOAI_PSEUDONYM_SECRET": "use-outro-segredo-longo-e-estavel-para-pseudonimos",
            },
            clear=False,
        ):
            with self.assertRaises(RuntimeError):
                AuthManager.from_env()


class AuditStoreTests(unittest.TestCase):
    def test_audit_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AuditStore(Path(tmp) / "audit.sqlite3")
            store.log(
                actor="clinician",
                role="admin",
                action="analysis.start",
                resource_type="job",
                resource_id="job-1",
                metadata={"movement": "Flexão de Cotovelo"},
            )
            rows = store.recent(10)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["actor"], "clinician")
            self.assertEqual(rows[0]["metadata"]["movement"], "Flexão de Cotovelo")

    def test_redact_patient_preserves_event_without_pseudonym(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AuditStore(Path(tmp) / "audit.sqlite3")
            store.log(
                actor="admin", role="admin", action="analysis.start",
                resource_type="job", resource_id="job-1",
                metadata={"patient": "P-ABC", "nested": {"patient": "P-ABC"}},
            )
            self.assertEqual(store.redact_patient("P-ABC"), 1)
            event = store.recent(1)[0]
            self.assertEqual(event["metadata"]["patient"], "[REDACTED]")
            self.assertEqual(event["metadata"]["nested"]["patient"], "[REDACTED]")


if __name__ == "__main__":
    unittest.main()
