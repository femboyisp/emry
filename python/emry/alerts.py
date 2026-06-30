"""Opt-in webhook alerting for non-finite metrics.

When a metric goes ``NaN``/``Inf`` — the clearest sign a run has gone wrong —
[`WebhookNotifier`] POSTs a Slack-compatible ``{"text": ...}`` message to a
configured URL (Slack, Discord, and most generic webhooks accept this shape).

Design constraints (matching the rest of Emry): never block or break the
training loop. The POST runs on a short-lived daemon thread with a timeout, each
metric alerts at most once per run, and any failure is swallowed. Pure stdlib —
no dependency — so it works in every deploy mode.
"""

from __future__ import annotations

import json
import math
import threading
import urllib.request
from typing import Any, Callable, Mapping, Optional

__all__ = ["WebhookNotifier"]


def _post(url: str, payload: dict) -> None:
    """POSTs `payload` as JSON to `url` (best-effort; short timeout)."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310 - user-supplied https webhook URL
        url, data=data, headers={"Content-Type": "application/json"}
    )
    urllib.request.urlopen(req, timeout=5).close()  # noqa: S310


class WebhookNotifier:
    """Fires a one-shot webhook the first time each metric goes non-finite."""

    def __init__(
        self,
        url: str,
        *,
        project: str = "",
        post: Callable[[str, dict], None] = _post,
        threaded: bool = True,
    ) -> None:
        self._url = url
        self._project = project
        self._post = post
        self._threaded = threaded
        self._alerted: set[str] = set()

    def check(self, values: Mapping[str, float], step: int) -> None:
        """Alerts (once per metric) for any non-finite value in `values`."""
        for name, value in values.items():
            if name not in self._alerted and not math.isfinite(value):
                self._alerted.add(name)
                self._fire(name, value, step)

    def _fire(self, name: str, value: float, step: int) -> None:
        prefix = f"{self._project}: " if self._project else ""
        payload = {"text": f"⚠️ emry {prefix}metric `{name}` is {value} at step {step}"}
        if self._threaded:
            # Detach the POST so a slow/unreachable webhook never stalls the loop.
            threading.Thread(target=self._safe_post, args=(payload,), daemon=True).start()
        else:
            self._safe_post(payload)

    def _safe_post(self, payload: dict) -> None:
        try:
            self._post(self._url, payload)
        except Exception:  # noqa: BLE001 - alerting must never break the run
            pass


def resolve(alert_webhook: Any, *, project: str) -> Optional[WebhookNotifier]:
    """Resolves the `alert_webhook` option to a notifier or `None`.

    Accepts a URL string, an already-built `WebhookNotifier` (for tests), or
    `None` to disable.
    """
    if alert_webhook is None:
        return None
    if isinstance(alert_webhook, WebhookNotifier):
        return alert_webhook
    if isinstance(alert_webhook, str) and alert_webhook:
        return WebhookNotifier(alert_webhook, project=project)
    return None
