import base64
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from libs.settings import Settings


class ApiKeySettingsTest(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows DPAPI")
    def test_encrypted_key_round_trip_from_disk(self):
        key = "synthetic-vision-key-0514"
        with tempfile.TemporaryDirectory() as root:
            settings = Settings()
            settings.path = str(Path(root) / "settings.pkl")
            settings["llm_model"] = "synthetic-vision"
            settings.set_api_key("  " + key + "  ")
            settings.save()
            data = Path(settings.path).read_bytes()
            for plaintext in (key.encode(), key.encode("utf-16-le"), base64.b64encode(key.encode())):
                self.assertNotIn(plaintext, data)
            restored = Settings()
            restored.path = settings.path
            self.assertTrue(restored.load())
            self.assertEqual(restored.get_api_key(), key)
            self.assertEqual(restored.get("llm_model"), "synthetic-vision")
            self.assertNotIn("llm_api_key", restored.data)

    def test_clear_removes_stored_credentials_without_encryption(self):
        settings = Settings()
        settings.data = {"llm_api_key_dpapi": "old-encrypted", "llm_api_key": "legacy", "llm_model": "vision"}
        with patch("libs.settings._crypt_api_key") as crypt:
            settings.set_api_key("  ")
            crypt.assert_not_called()
        self.assertEqual(settings.data, {"llm_model": "vision"})
        self.assertEqual(settings.get_api_key(), "")

    def test_encryption_failure_keeps_previous_value_and_never_stores_plaintext(self):
        settings = Settings()
        settings.data = {"llm_api_key_dpapi": "previous-encrypted"}
        with patch("libs.settings._crypt_api_key", side_effect=OSError("test failure")):
            with self.assertRaises(OSError):
                settings.set_api_key("synthetic-new-key")
        self.assertEqual(settings.data, {"llm_api_key_dpapi": "previous-encrypted"})

    def test_bad_ciphertext_returns_empty_without_logging_the_value(self):
        for encrypted in ("not valid base64!", base64.b64encode(b"not a DPAPI blob").decode("ascii")):
            with self.subTest(encrypted=encrypted):
                settings = Settings()
                settings["llm_api_key_dpapi"] = encrypted
                with self.assertLogs("PPOCRLabel", level="WARNING") as logs:
                    self.assertEqual(settings.get_api_key(), "")
                self.assertNotIn(encrypted, "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
