from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.e2e_smoke import multipart_body, ws_url


class E2EHelperTests(unittest.TestCase):
    def test_ws_url_preserves_host_and_switches_scheme(self):
        self.assertEqual(
            ws_url("http://localhost:8080", "/ws/analise/abc"),
            "ws://localhost:8080/ws/analise/abc",
        )
        self.assertEqual(
            ws_url("https://example.test", "/ws/analise/abc"),
            "wss://example.test/ws/analise/abc",
        )

    def test_multipart_contains_fields_and_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "sample.mp4"
            video.write_bytes(b"video-bytes")
            body, content_type = multipart_body(
                {"movimento": "Flexão de Cotovelo", "lado": "direito"},
                "video_a",
                video,
            )
        self.assertIn("multipart/form-data; boundary=", content_type)
        self.assertIn(b'content-disposition: form-data; name="movimento"'.lower(), body.lower())
        self.assertIn("Flexão de Cotovelo".encode("utf-8"), body)
        self.assertIn(b'filename="sample.mp4"', body)
        self.assertIn(b"video-bytes", body)


if __name__ == "__main__":
    unittest.main()
