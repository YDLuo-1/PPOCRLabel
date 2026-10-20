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
    for region, annotation in enumerate(annotations, 1):
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
        for point_index, point in enumerate(points, 1):
            if (
                not isinstance(point, list)
                or len(point) != 2
                or any(
                    isinstance(n, bool)
                    or not isinstance(n, (int, float))
                    or not math.isfinite(n)
                    for n in point
                )
            ):
                raise InvalidAnnotationError(
                    "Annotation %s point %s must be two finite numbers; image is %sx%s."
                    % (region, point_index, width, height)
                )
            if not (0 <= point[0] <= width and 0 <= point[1] <= height):
                raise InvalidAnnotationError(
                    "Annotation %s point %s (%s, %s) is outside the %sx%s image; "
                    "use 0 <= x <= %s and 0 <= y <= %s in original-image pixels."
                    % (
                        region,
                        point_index,
                        point[0],
                        point[1],
                        width,
                        height,
                        width,
                        height,
                    )
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
            "Detect the actual text boundaries and transcribe the text inside each boundary.\n"
            "Coordinate rules:\n"
            "- Use original-image pixel coordinates. The origin (0,0) is the image's "
            "top-left corner; x increases rightward and y increases downward. "
            f"Every point must satisfy 0 <= x <= {width} and 0 <= y <= {height}. "
            "Do not use percentages, a normalized 0-1 or 0-1000 scale, screen coordinates, "
            "or coordinates from a resized or cropped view.\n"
            "- Return four distinct integer [x,y] points for each text line, in clockwise "
            "order: top-left, top-right, bottom-right, bottom-left. The quadrilateral "
            "must have nonzero area and follow the text's orientation.\n"
            "Text boundary rules:\n"
            "- Locate the outermost visible strokes before choosing the four corners. "
            "The polygon must enclose ALL visible strokes of ALL characters in that line, "
            "including the first and last characters, punctuation, symbols, accents, "
            "and strokes extending above or below the main character body.\n"
            "- Keep the polygon tight with only a small outer margin. No edge may "
            "cut through a visible character or leave its top, bottom, left, or right "
            "strokes outside. Do not return a box covering only the middle or bottom "
            "of a line. Do not include large areas of unrelated background.\n"
            "- Use the text itself as the boundary, not surrounding borders, grooves, "
            "table cells, decorations, shadows, or background lines. Locate each line "
            "independently; keep separate lines or spatially separate text regions "
            "in separate polygons and do not duplicate a region.\n"
            "Transcription rules:\n"
            "- Read only visible characters. Preserve case and punctuation unless "
            "additional project instructions specify a transcription convention. "
            "Do not invent or complete obscured or off-image characters from context "
            "or neighboring text. Use ### for an unreadable text region.\n"
            "Before returning, visually recheck every polygon against the image: "
            "all corresponding visible strokes must be inside it, with no clipping "
            "or offset onto the background. Check the top and bottom edges as well "
            "as both ends of the line.\n"
            "Return JSON only, without Markdown, comments, or explanations, in this format: "
            '{"annotations":[{"transcription":"visible text","points":[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}]}. '
            'If there is no target text, return {"annotations":[]}. '
            "Additional project instructions may define which text to annotate and "
            "how to transcribe or ignore it; keep the JSON format and coordinate rules."
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
        for attempt in range(2):
            content = self._request(payload, cancelled)
            try:
                return parse_annotations(content, width, height)
            except InvalidAnnotationError as error:
                if attempt:
                    raise InvalidAnnotationError(
                        "%s The single correction retry also failed." % error
                    ) from None
                # Keep each request self-contained for providers with different
                # requirements for assistant/reasoning history.
                payload["messages"][0]["content"][0]["text"] += (
                    "\nAn earlier annotation attempt failed validation: %s\n"
                    "Recheck the SAME original %sx%s image and return corrected JSON only. "
                    "Every point must be a numeric [x,y] pair in original-image PIXELS: "
                    "0 <= x <= %s, 0 <= y <= %s. Do not use a 0-1000 scale or coordinates "
                    "of a resized image. Do not invent text outside the image."
                ) % (error, width, height, width, height)

    def _request(self, payload, cancelled):
        if cancelled and cancelled():
            raise AnnotationCancelled()
        if cancelled is None:
            return self._complete(payload)
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
        return content
