import base64
import json
import re
import time

import requests

from config import debug_print

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# A 429 here means this account's per-minute token budget for this model
# is exhausted, not that the request itself was bad -- confirmed via a
# real 429's body ("Please try again in 18.57s") and its "retry-after"
# header, which Groq sets to the actual wait needed. Retried once (not
# looped indefinitely): a second consecutive 429 means something other
# than routine quota timing (e.g. sustained heavy concurrent use), which
# piling on more waits won't fix. A hard cap on top of whatever Groq
# reports guards against ever blocking a page for an implausibly long time
# if that header is ever missing or wrong.
MAX_RATE_LIMIT_WAIT_SECONDS = 65

# Below this many tokens of headroom (per Groq's own x-ratelimit-
# remaining-tokens header from the previous call), wait out the reset
# window *before* even attempting the next call, rather than firing it
# and getting a 429 back -- confirmed on a real batch run (this account's
# quota is 8000 tokens/minute) that a real batched call costs ~2200-2500
# tokens, so a call attempted with less than that left is a near-certain
# 429; pre-emptively waiting here trades a smaller, predictable pause for
# the unpredictable/larger reactive one _RateLimited otherwise causes,
# and skips the wasted round-trip of a request that was always going to
# fail.
PACING_TOKEN_THRESHOLD = 3000
DEFAULT_PACING_WAIT_SECONDS = 20.0

# Groq's reset-window headers come as e.g. "18.15s" or "8m38.4s".
_RESET_DURATION = re.compile(r"(?:(\d+)m)?([\d.]+)s")


def _parse_reset_seconds(value):
    if not value:
        return None
    match = _RESET_DURATION.match(value.strip())
    if not match:
        return None
    minutes = float(match.group(1)) if match.group(1) else 0.0
    return minutes * 60 + float(match.group(2))

_BATCH_RECHECK_PROMPT = (
    "Below is a list of {count} rows to re-examine on this page, each "
    "identified by its Date column value (and sometimes a disambiguating "
    "hint, when more than one row shares the same date). For EACH row "
    "listed, look ONLY at that row and report its In time and Out time "
    "cell values exactly as written, 24-hour HH:MM format. If a value is "
    "blank or illegible, use an empty string for it.\n\n"
    "Rows to check:\n{row_list}\n\n"
    'Return ONLY a JSON object, no other text, shaped exactly like this: '
    '{{"results": [{{"in_time": "...", "out_time": "..."}}, ...]}} -- '
    "one entry per row above, in the exact same order, nothing added or "
    "removed."
)

# Groq's json_object mode (unlike Mistral's document_annotation_format)
# doesn't enforce a schema, so the model occasionally wraps the JSON in a
# markdown code fence despite the prompt -- stripped out before parsing.
_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


# Groq's error body names which budget was hit ("tokens per minute (TPM)"
# vs "tokens per day (TPD)"). Only the per-minute one is worth a reactive
# wait-and-retry -- confirmed via a real run that the daily one's own
# retry-after routinely comes back in the 15-20+ *minute* range, so
# capping and waiting MAX_RATE_LIMIT_WAIT_SECONDS then retrying is a
# near-certain second failure, burning ~65s per page for nothing across
# every remaining page once the daily budget is actually exhausted.
_DAILY_QUOTA_MARKER = "tokens per day"


class _RateLimited(Exception):
    def __init__(self, message, retry_after, is_daily=False):
        super().__init__(message)
        self.retry_after = retry_after
        self.is_daily = is_daily


class VisionRecheckEngine:
    """
    Second opinion on flagged rows, using a genuinely different model
    (Qwen 3.6 27B, via Groq) from the one that produced the original read
    (Mistral's Document AI OCR, see mistral_ocr_engine.py). Mistral
    re-checking its own read only catches disagreements when its second
    pass happens to land differently -- it's blind to a misread Mistral
    reads the same confident (wrong) way every time. An independent
    model is more likely to actually disagree on those cells.

    This project's second-opinion model has changed twice: Llama 4 Scout
    (Groq) originally, then Groq removed every vision-capable Llama model
    from this account's catalog outright (confirmed via GET /v1/models --
    none left at all, not just a renamed one), so a genuinely different
    replacement had to be found rather than just updating a model string.
    gpt-oss-120b (also on Groq) was tried first but rejected outright --
    confirmed via a direct API call that it's text-only ("messages[0]
    .content must be a string" on an image_url payload) and can't see the
    page at all, which would make this "recheck" just Mistral agreeing
    with itself over a text relay. Qwen 3.6 27B (Groq) was then confirmed
    (same direct-call method) to actually accept image_url content and
    return a legible read of real handwriting, so it's what's wired in
    now -- keep that verification method in mind if this model also stops
    working someday rather than assuming a 404 means a renamed model.

    Batches every flagged row on a page into ONE call (recheck_rows), not
    one call per row -- confirmed on a real run that this account's Groq
    quota for this model is only 8000 tokens/minute, and the page image
    itself (re-uploaded on every call) costs the large majority of that
    budget on its own, so a call per row per field burned through the
    whole minute's budget after just 1-2 calls and 429'd on nearly
    everything else. Uploading the image once per page and asking about
    every flagged row in that single request cuts token usage by roughly
    however many rows/fields would otherwise have triggered separate
    calls.

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

    def __init__(self, api_key, model="qwen/qwen3.6-27b"):
        self.api_key = api_key
        self.model = model
        # Carried across calls on this same instance (one instance lives
        # for a whole pipeline run, see pipeline.py) so pacing can react
        # to how much budget the *previous* page's call actually left,
        # not just this one's own retries.
        self._remaining_tokens = None
        self._reset_seconds = None
        # Set once a 429 names the *daily* budget (not the per-minute
        # one) as exhausted. Unlike the per-minute case, that budget
        # won't recover mid-run, so every subsequent page skips the call
        # entirely instead of repeating a guaranteed-failing round trip.
        self._daily_quota_blocked_until = None

    def recheck_rows(self, image_path, row_requests):
        """
        Re-examines every row in row_requests as an independent second
        opinion, in a single call against the page image.

        row_requests: list of (date, disambiguator) tuples -- disambiguator
        may be None. Order matters: the model is asked to answer in the
        same order, and results are matched back positionally, not by
        trusting the model's own bookkeeping.

        Returns a list of {"in_time": ..., "out_time": ...} dicts, the
        same length and order as row_requests, or None if the whole
        call fails or the response can't be matched back 1:1 to what was
        asked -- treating the whole page's recheck as unavailable is
        safer than guessing which answer belongs to which row.
        """
        if not row_requests:
            return []

        if (
            self._daily_quota_blocked_until is not None
            and time.time() < self._daily_quota_blocked_until
        ):
            debug_print(
                "VisionRecheckEngine: daily token quota still exhausted "
                f"(~{self._daily_quota_blocked_until - time.time():.0f}s "
                "left on Groq's own estimate), skipping recheck for this "
                "page rather than repeating a call that's certain to 429"
            )
            return None

        if (
            self._remaining_tokens is not None
            and self._remaining_tokens < PACING_TOKEN_THRESHOLD
        ):
            wait_seconds = min(
                self._reset_seconds or DEFAULT_PACING_WAIT_SECONDS,
                MAX_RATE_LIMIT_WAIT_SECONDS,
            )
            debug_print(
                f"VisionRecheckEngine: pacing -- only "
                f"{self._remaining_tokens} tokens left in this window, "
                f"waiting {wait_seconds:.1f}s before calling"
            )
            time.sleep(wait_seconds)

        for attempt in (1, 2):
            try:
                return self._call(image_path, row_requests)
            except _RateLimited as e:
                if e.is_daily:
                    self._daily_quota_blocked_until = time.time() + e.retry_after
                    debug_print(
                        "VisionRecheckEngine: daily token quota exhausted "
                        f"(not per-minute -- won't recover this run), "
                        f"skipping retry and pausing recheck for "
                        f"~{e.retry_after:.0f}s: {e}"
                    )
                    return None
                if attempt == 2:
                    debug_print(
                        f"VisionRecheckEngine: batch recheck rate-limited "
                        f"again after waiting, giving up: {e}"
                    )
                    return None
                wait_seconds = min(e.retry_after, MAX_RATE_LIMIT_WAIT_SECONDS)
                debug_print(
                    f"VisionRecheckEngine: rate-limited, waiting "
                    f"{wait_seconds:.1f}s before one retry: {e}"
                )
                time.sleep(wait_seconds)
            except Exception as e:
                debug_print(f"VisionRecheckEngine: batch recheck failed: {e}")
                return None

    def _call(self, image_path, row_requests):
        """
        One attempt at the batched recheck call. Raises _RateLimited on a
        429 (caller decides whether/how long to wait and retry); any
        other failure is left to propagate to the caller's own handling.
        """
        b64, ext = self._encode_image(image_path)

        row_list = "\n".join(
            f'{i}. Date "{date}"' + (f", {disambiguator}" if disambiguator else "")
            for i, (date, disambiguator) in enumerate(row_requests, start=1)
        )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": _BATCH_RECHECK_PROMPT.format(
                                count=len(row_requests), row_list=row_list
                            ),
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
            # Qwen's chain-of-thought reasoning otherwise competes with
            # the actual JSON answer for output tokens -- confirmed on
            # a real batched call that it ate the whole response
            # budget and left Groq's own JSON-schema validation with
            # nothing to validate (400 json_validate_failed, empty
            # failed_generation). Turning it off also roughly a
            # thirds the token cost per call (a real call measured
            # ~7000 tokens with reasoning on vs. ~2200 with it off),
            # which matters directly for this model's tight per-
            # minute quota on this account.
            "reasoning_effort": "none",
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=120)

        # Captured regardless of status -- a 429 response carries these
        # same headers (showing the budget already at/near zero), just as
        # useful for pacing the *next* call as a successful response's.
        remaining = resp.headers.get("x-ratelimit-remaining-tokens")
        if remaining is not None:
            try:
                self._remaining_tokens = int(remaining)
            except ValueError:
                pass
        self._reset_seconds = _parse_reset_seconds(resp.headers.get("x-ratelimit-reset-tokens"))

        if resp.status_code == 429:
            retry_after = resp.headers.get("retry-after")
            body = resp.text
            raise _RateLimited(
                body[:300],
                retry_after=float(retry_after) if retry_after else 20.0,
                is_daily=_DAILY_QUOTA_MARKER in body.lower(),
            )

        resp.raise_for_status()

        message = resp.json()["choices"][0]["message"]["content"]
        cleaned = _JSON_FENCE.sub("", message.strip())
        parsed = json.loads(cleaned)
        results = parsed.get("results", [])

        if len(results) != len(row_requests):
            debug_print(
                f"VisionRecheckEngine: batch recheck returned "
                f"{len(results)} results for {len(row_requests)} rows "
                "requested, discarding (can't match positionally)"
            )
            return None

        return [
            {
                "in_time": r.get("in_time", "") or "",
                "out_time": r.get("out_time", "") or "",
            }
            for r in results
        ]

    def _encode_image(self, image_path):
        with open(image_path, "rb") as f:
            data = f.read()
        ext = image_path.rsplit(".", 1)[-1].lower() or "jpeg"
        return base64.b64encode(data).decode("utf-8"), ext
