import base64
import json
import re

import requests

from config import debug_print

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

_RECHECK_PROMPT = (
    'Look ONLY at the row whose Date column reads "{date}"{block_hint}. '
    "Re-read that row very carefully and report its In time and Out time "
    "cell values exactly as written, 24-hour HH:MM format. If a value is "
    "blank or illegible, use an empty string for it.\n\n"
    'Return ONLY a JSON object, no other text: {{"in_time": "...", "out_time": "..."}}'
)

# Groq's json_object mode (unlike Mistral's document_annotation_format)
# doesn't enforce a schema, so the model occasionally wraps the JSON in a
# markdown code fence despite the prompt -- stripped out before parsing.
_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


class LlamaVisionEngine:
    """
    Second opinion on flagged rows, using a genuinely different model
    (Llama 4 Scout, via Groq) from the one that produced the original
    read (Mistral's Document AI OCR, see mistral_ocr_engine.py). Mistral
    re-checking its own read only catches disagreements when its second
    pass happens to land differently -- it's blind to a misread Mistral
    reads the same confident (wrong) way every time. An independent
    model is more likely to actually disagree on those cells.

    Sends the *same full page image* as the original read, not an
    isolated cell crop -- there's no per-cell bounding-box geometry
    available now that Surya's table_rec grid is gone (see
    mistral_ocr_engine.py docstring), and the prior cell-crop approach
    (see git history: llama_vision_verifier.py on the llama-vision-
    integration branch) depended entirely on that grid to know where to
    cut. Whole-row/whole-page context also gave the model row-tracking
    cues a bare cell crop doesn't have.

    Which rows get rechecked at all, and what counts as "disagreement",
    are engine-agnostic and live in recheck_utils.py, shared with
    whatever the caller uses as the recheck engine -- so swapping this
    class in for a different one doesn't require touching the gating
    logic.
    """

    def __init__(self, api_key, model="meta-llama/llama-4-scout-17b-16e-instruct"):
        self.api_key = api_key
        self.model = model

    def recheck_row(self, image_path, date, disambiguator=None):
        """
        Re-examines a single row as an independent second opinion.
        Returns {"in_time": ..., "out_time": ...} or None if the call
        fails or the response isn't usable.

        disambiguator: see MistralOCREngine.recheck_row -- same shape
        of hint (block position and/or "Nth row with this date"), used
        identically here since a duplicate date is just as ambiguous to
        this model as it is to Mistral's.
        """
        try:
            b64, ext = self._encode_image(image_path)
            hint_text = f", {disambiguator}" if disambiguator else ""

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": _RECHECK_PROMPT.format(date=date, block_hint=hint_text),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/{ext};base64,{b64}"},
                            },
                        ],
                    }
                ],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()

            message = resp.json()["choices"][0]["message"]["content"]
            cleaned = _JSON_FENCE.sub("", message.strip())
            result = json.loads(cleaned)

            return {
                "in_time": result.get("in_time", "") or "",
                "out_time": result.get("out_time", "") or "",
            }
        except Exception as e:
            debug_print(f"LlamaVisionEngine: recheck_row failed for {date}: {e}")
            return None

    def _encode_image(self, image_path):
        with open(image_path, "rb") as f:
            data = f.read()
        ext = image_path.rsplit(".", 1)[-1].lower() or "jpeg"
        return base64.b64encode(data).decode("utf-8"), ext
