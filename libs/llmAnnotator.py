"""Whole-image OCR through a vision-capable Chat Completions endpoint."""

import base64
import json
import math
import os
import queue
import re
import threading
from urllib.parse import urlsplit

import cv2
import requests


class LLMServiceError(Exception):
    """A service/configuration error that stops the batch."""


class InvalidAnnotationError(ValueError):
    """An image failed; its existing annotations must be preserved."""


class AnnotationCancelled(Exception):
    pass


def service_error_detail(response, api_key=""):
    """Show the provider's structured error without echoing credentials/images."""
    try:
        body = response.json()
    except ValueError:
        return ""
    if not isinstance(body, dict):
        return ""
    error = body.get("error", body)
    if isinstance(error, str):
        message = error
    elif isinstance(error, dict):
        message = error.get("message", "")
        if not isinstance(message, str):
            return ""
        for key in ("code", "param"):
            value = error.get(key)
            if isinstance(value, (str, int)) and not isinstance(value, bool):
                message += " (%s: %s)" % (key, str(value)[:128])
    else:
        return ""
    if api_key:
        message = message.replace(api_key, "[redacted]")
    message = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[redacted]", message)
    message = re.sub(r"(?i)\bBearer\s+[^\s,\"']+", "Bearer [redacted]", message)
    message = re.sub(r"https?://[^\s\"']+", "[URL]", message)
    message = re.sub(r"data:image/[^\s\"']+", "[image data]", message)
    return " ".join(message.split())[:800]


def service_endpoint(base_url, model):
    parsed = urlsplit(base_url.strip())
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "Enter an HTTP(S) API base URL without credentials or query parameters."
        )
    if not model.strip():
        raise ValueError("Enter the name of a vision-capable model.")
    endpoint = base_url.strip().rstrip("/")
    return (
        endpoint
        if endpoint.endswith("/chat/completions")
        else endpoint + "/chat/completions"
    )


def parse_annotations(content, width, height):
    if not isinstance(content, str):
        raise InvalidAnnotationError("The model did not return text containing JSON.")
    text = content.strip()
    fence = chr(96) * 3
    if text.startswith(fence):
        lines = text.splitlines()
        if len(lines) < 3 or lines[-1].strip() != fence:
            raise InvalidAnnotationError("Incomplete JSON code block.")
        text = "\n".join(lines[1:-1])
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        raise InvalidAnnotationError("The model returned invalid JSON.") from None
    annotations = value.get("annotations") if isinstance(value, dict) else None
    if not isinstance(annotations, list):
        raise InvalidAnnotationError("Expected an annotations array.")
    result = []
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise InvalidAnnotationError("Each annotation must be an object.")
        transcription, points = annotation.get("transcription"), annotation.get(
            "points"
        )
        if not isinstance(transcription, str) or not transcription.strip():
            raise InvalidAnnotationError(
                "Each annotation needs non-empty transcription."
            )
        if not isinstance(points, list) or len(points) != 4:
            raise InvalidAnnotationError("Each annotation needs four polygon points.")
        polygon = []
        for point in points:
            if (
                not isinstance(point, list)
                or len(point) != 2
                or any(
                    isinstance(n, bool)
                    or not isinstance(n, (int, float))
                    or not math.isfinite(n)
                    for n in point
                )
                or not (0 <= point[0] <= width and 0 <= point[1] <= height)
            ):
                raise InvalidAnnotationError(
                    "Polygon coordinates are invalid or outside the image."
                )
            polygon.append([int(round(point[0])), int(round(point[1]))])
        crosses = []
        for i in range(4):
            a, b, c = polygon[i], polygon[(i + 1) % 4], polygon[(i + 2) % 4]
            crosses.append(
                (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
            )
        if not (all(n > 0 for n in crosses) or all(n < 0 for n in crosses)):
            raise InvalidAnnotationError(
                "Expected a non-degenerate convex quadrilateral."
            )
        result.append([polygon, (transcription, None)])
    return result


class LLMAnnotator:
    def __init__(self, base_url, model, api_key="", instructions=""):
        self.endpoint = service_endpoint(base_url, model)
        self.model = model.strip()
        self.api_key = (
            api_key.strip() or os.environ.get("PPOCRLABEL_API_KEY", "").strip()
        )
        self.instructions = instructions

    def _complete(self, payload):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        try:
            with requests.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=(10, 120),
                allow_redirects=False,
            ) as response:
                if not 200 <= response.status_code < 300:
                    detail = service_error_detail(response, self.api_key)
                    message = "Vision service returned HTTP %s." % response.status_code
                    if detail:
                        message += " " + detail
                if response.status_code in (413, 422):
                    raise InvalidAnnotationError(message)
                if not 200 <= response.status_code < 300:
                    raise LLMServiceError(message)
                if len(response.content) > 8 * 1024 * 1024:
                    raise InvalidAnnotationError("Vision response is too large.")
                try:
                    content = response.json()["choices"][0]["message"]["content"]
                    if isinstance(content, list):
                        content = "".join(
                            part["text"]
                            for part in content
                            if part.get("type") == "text"
                        )
                    return content
                except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                    raise InvalidAnnotationError(
                        "Invalid Chat Completions response."
                    ) from None
        except requests.exceptions.ConnectTimeout:
            raise LLMServiceError(
                "Connection to the vision service timed out."
            ) from None
        except requests.exceptions.ReadTimeout:
            raise InvalidAnnotationError(
                "Vision request timed out (120 seconds)."
            ) from None
        except requests.exceptions.RequestException:
            # Do not log provider responses, headers or URLs containing secrets.
            raise LLMServiceError("Could not connect to the vision service.") from None

    def annotate(self, image, cancelled=None):
        if cancelled and cancelled():
            raise AnnotationCancelled()
        height, width = image.shape[:2]
        success, encoded = cv2.imencode(".png", image)
        if not success:
            raise InvalidAnnotationError("Could not encode the image.")
        prompt = (
            "Annotate the visible text in this image for OCR training. "
            f"The original image is {width} pixels wide and {height} pixels high. "
            "Return JSON only in this format: "
            '{"annotations":[{"transcription":"visible text","points":[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}]}. '
            "Use original-image pixel coordinates, never normalized coordinates. "
            "Each quadrilateral must tightly enclose one text line, with points "
            "in clockwise order starting at the top left. Read only visible "
            "characters; do not invent obscured characters. Use ### for an "
            "unreadable text region. If there is no text, return "
            '{"annotations":[]}.'
        )
        if self.instructions.strip():
            prompt += (
                "\nAdditional annotation instructions:\n" + self.instructions.strip()
            )
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,"
                                + base64.b64encode(encoded).decode("ascii"),
                            },
                        },
                    ],
                }
            ],
        }
        if cancelled is None:
            content = self._complete(payload)
        else:
            # A cancelled HTTP call may finish remotely. Keep that bounded call
            # independent of Qt so its late response cannot touch UI or labels.
            completed = queue.Queue(maxsize=1)

            def request():
                try:
                    completed.put((self._complete(payload), None))
                except Exception as error:
                    completed.put((None, error))

            threading.Thread(target=request, daemon=True).start()
            while True:
                if cancelled():
                    raise AnnotationCancelled()
                try:
                    content, error = completed.get(timeout=0.1)
                    if error is not None:
                        raise error
                    break
                except queue.Empty:
                    continue
            if cancelled():
                raise AnnotationCancelled()
        return parse_annotations(content, width, height)
