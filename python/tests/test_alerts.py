"""Tests for the non-finite webhook notifier (POST injected, no network)."""

from emry import run
from emry.alerts import WebhookNotifier, resolve


def _notifier(project=""):
    """A synchronous notifier recording (url, payload) POSTs into a list."""
    posts = []
    n = WebhookNotifier(
        "http://hook",
        project=project,
        post=lambda url, payload: posts.append((url, payload)),
        threaded=False,
    )
    return n, posts


def test_alerts_only_on_non_finite():
    n, posts = _notifier()
    n.check({"loss": 1.0, "lr": 0.1}, step=0)
    assert posts == []  # all finite
    n.check({"loss": float("nan")}, step=5)
    assert len(posts) == 1
    url, payload = posts[0]
    assert url == "http://hook" and "loss" in payload["text"] and "step 5" in payload["text"]


def test_alerts_once_per_metric():
    n, posts = _notifier()
    n.check({"loss": float("inf")}, step=1)
    n.check({"loss": float("inf")}, step=2)  # same metric, already alerted
    n.check({"acc": float("nan")}, step=3)  # different metric -> a second alert
    assert len(posts) == 2


def test_post_failure_is_swallowed():
    def boom(url, payload):
        raise OSError("webhook down")

    n = WebhookNotifier("http://hook", post=boom, threaded=False)
    n.check({"loss": float("nan")}, step=0)  # must not raise


def test_resolve_accepts_url_instance_or_none():
    assert resolve(None, project="p") is None
    assert resolve("", project="p") is None
    assert isinstance(resolve("http://hook", project="p"), WebhookNotifier)
    existing = WebhookNotifier("http://hook")
    assert resolve(existing, project="p") is existing


def test_run_fires_webhook_on_nan(tmp_path):
    posts = []
    notifier = WebhookNotifier(
        "http://hook",
        project="p",
        post=lambda url, payload: posts.append(payload),
        threaded=False,
    )
    with run(
        "p",
        metrics=["loss"],
        mode="file",
        log_dir=str(tmp_path),
        gpu=False,
        alert_webhook=notifier,
    ) as r:
        r.emit(loss=1.0)  # finite, no alert
        r.emit(loss=float("nan"))  # triggers the webhook
    assert len(posts) == 1 and "loss" in posts[0]["text"]
