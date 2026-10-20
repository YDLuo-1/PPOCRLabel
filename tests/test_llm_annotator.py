import base64
import json
import os
import threading
import time
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import cv2
import numpy as np
import requests

from libs.llmAnnotator import (
    AnnotationCancelled,
    InvalidAnnotationError,
    LLMAnnotator,
    LLMServiceError,
    parse_annotations,
    service_endpoint,
)

ANNOTATIONS = {
    "annotations": [
        {
            "transcription": "A0514+",
            "points": [[2, 3], [110, 3], [110, 25], [2, 25]],
        }
    ]
}


def completion(annotations=ANNOTATIONS):
    return {"choices": [{"message": {"content": json.dumps(annotations)}}]}


@contextmanager
def vision_server(responder=None):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.server.received.append((self.path, body, dict(self.headers)))
            status, value = responder(body) if responder else (200, completion())
            data = json.dumps(value).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    server.received = []
    thread = threading.Thread(
        target=lambda: server.serve_forever(poll_interval=0.05), daemon=True
    )
    thread.start()
    try:
        yield "http://127.0.0.1:%s/v1" % server.server_port, server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class LLMAnnotatorTest(unittest.TestCase):
    def setUp(self):
        self.image = np.full((80, 120, 3), 200, dtype=np.uint8)

    def test_whole_image_request_and_direct_polygon_result(self):
        with vision_server() as (url, server):
            result = LLMAnnotator(
                url, "test-vision", "session-test-key", "Read visible serial numbers."
            ).annotate(self.image)
            path, body, headers = server.received[0]
            self.assertEqual(path, "/v1/chat/completions")
            self.assertEqual(headers["Authorization"], "Bearer session-test-key")
            self.assertEqual(body["model"], "test-vision")
            self.assertNotIn("response_format", body)
            content = body["messages"][0]["content"]
            self.assertIn("120 pixels wide and 80 pixels high", content[0]["text"])
            self.assertIn("Read visible serial numbers.", content[0]["text"])
            data = content[1]["image_url"]["url"].split(",", 1)[1]
            decoded = cv2.imdecode(
                np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_COLOR
            )
            np.testing.assert_array_equal(decoded, self.image)
            self.assertEqual(result[0][0], ANNOTATIONS["annotations"][0]["points"])
            self.assertEqual(result[0][1], ("A0514+", None))

    def test_empty_and_unreadable_annotations_and_fenced_json(self):
        self.assertEqual(parse_annotations('{"annotations":[]}', 120, 80), [])
        value = json.loads(json.dumps(ANNOTATIONS))
        value["annotations"][0]["transcription"] = "###"
        fence = chr(96) * 3
        result = parse_annotations(
            fence + "json\n" + json.dumps(value) + "\n" + fence, 120, 80
        )
        self.assertEqual(result[0][1], ("###", None))

    def test_invalid_output_is_rejected_as_a_whole_image(self):
        values = ["broken", "[]", "{}", '{"annotations":null}']
        invalid_polygons = [
            [[-1, 0], [10, 0], [10, 10], [0, 10]],
            [[0, 0], [121, 0], [121, 10], [0, 10]],
            [[0, 0], [10, 10], [0, 10], [10, 0]],
            [[0, 0], [10, 0], [10, 0], [0, 0]],
            [[True, 0], [10, 0], [10, 10], [0, 10]],
            [[float("nan"), 0], [10, 0], [10, 10], [0, 10]],
            [[float("inf"), 0], [10, 0], [10, 10], [0, 10]],
        ]
        for points in invalid_polygons:
            values.append(
                json.dumps(
                    {
                        "annotations": [
                            ANNOTATIONS["annotations"][0],
                            {
                                "transcription": "bad",
                                "points": points,
                            },
                        ]
                    }
                )
            )
        values.append(
            json.dumps({"annotations": [{"transcription": "", "points": []}]})
        )
        for value in values:
            with self.subTest(value=value):
                with self.assertRaises(InvalidAnnotationError):
                    parse_annotations(value, 120, 80)

    def test_service_errors_stop_without_leaking_response_details(self):
        for status in (401, 403, 404, 429, 503):
            with self.subTest(status=status):
                with vision_server(
                    lambda body: (status, {"secret": "provider-private-value"})
                ) as (url, server):
                    with self.assertRaises(LLMServiceError) as caught:
                        LLMAnnotator(url, "vision").annotate(self.image)
                    self.assertNotIn("provider-private-value", str(caught.exception))
                    self.assertEqual(len(server.received), 1)

    def test_image_rejection_and_malformed_envelope(self):
        for status, value in [(413, {}), (422, {}), (200, {"choices": []})]:
            with self.subTest(status=status):
                with vision_server(lambda body: (status, value)) as (url, _):
                    with self.assertRaises(InvalidAnnotationError):
                        LLMAnnotator(url, "vision").annotate(self.image)

    def test_connection_and_read_timeouts_are_bounded_and_sanitized(self):
        annotator = LLMAnnotator("http://127.0.0.1:1/v1", "vision")
        for error, expected in [
            (requests.exceptions.ConnectTimeout("private"), LLMServiceError),
            (requests.exceptions.ConnectionError("private"), LLMServiceError),
            (requests.exceptions.ReadTimeout("private"), InvalidAnnotationError),
        ]:
            with patch("libs.llmAnnotator.requests.post", side_effect=error) as post:
                with self.assertRaises(expected) as caught:
                    annotator.annotate(self.image)
                self.assertNotIn("private", str(caught.exception))
                self.assertEqual(post.call_args.kwargs["timeout"], (10, 120))
                self.assertFalse(post.call_args.kwargs["allow_redirects"])

    def test_cancel_discards_inflight_response_promptly(self):
        entered, release, cancelled = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )

        def delayed(body):
            entered.set()
            release.wait(3)
            return 200, completion()

        def cancel():
            entered.wait(2)
            cancelled.set()

        with vision_server(delayed) as (url, server):
            threading.Thread(target=cancel, daemon=True).start()
            started = time.monotonic()
            try:
                with self.assertRaises(AnnotationCancelled):
                    LLMAnnotator(url, "vision").annotate(self.image, cancelled.is_set)
                self.assertLess(time.monotonic() - started, 1.5)
                self.assertEqual(len(server.received), 1)
            finally:
                release.set()

    def test_environment_key_and_endpoint_validation(self):
        with patch.dict(os.environ, {"PPOCRLABEL_API_KEY": "env-test-key"}):
            self.assertEqual(
                LLMAnnotator("http://localhost:8000/v1", "vision").api_key,
                "env-test-key",
            )
        self.assertEqual(
            service_endpoint("http://localhost/v1/chat/completions", "vision"),
            "http://localhost/v1/chat/completions",
        )
        for url in (
            "file:///tmp/service",
            "https://user:secret@example.test/v1",
            "https://example.test/v1?key=secret",
        ):
            with self.assertRaises(ValueError):
                service_endpoint(url, "vision")


if __name__ == "__main__":
    unittest.main()
