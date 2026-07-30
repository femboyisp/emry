# syntax=docker/dockerfile:1

# ---- Builder ---------------------------------------------------------------
# Pin to the workspace MSRV (see rust-toolchain.toml). The `emry` binary is
# produced by the `emry-cli` crate.
FROM rust:1.97-slim AS builder

WORKDIR /src

# Cache-friendly layering: copy only the manifests first so `cargo fetch`
# (dependency resolution + download) is cached until Cargo.toml/Cargo.lock
# actually change, not on every source edit.
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
# The Python SDK is part of the workspace's file tree but not needed to build
# the CLI; the manifests above reference only the Rust crates.

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    cargo fetch

# Now build the release binary. A cache mount on the target dir keeps
# incremental artifacts across builds; we copy the binary out afterwards.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release -p emry-cli \
    && cp /src/target/release/emry /usr/local/bin/emry

# ---- Runtime ---------------------------------------------------------------
# Debian slim (NOT distroless): the GPU poller and other helpers may shell out
# (e.g. `nvidia-smi`), so we keep a real shell and libc. Still small.
FROM debian:bookworm-slim AS runtime

# ca-certificates is handy for any outbound TLS (webhook/Slack alerts); tini as
# a tiny init so signals (SIGTERM on pod stop) are forwarded cleanly.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates tini \
    && rm -rf /var/lib/apt/lists/*

# Non-root user. Fixed UID/GID so a mounted /logs volume can be chowned to it.
RUN groupadd --gid 10001 emry \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin emry

COPY --from=builder /usr/local/bin/emry /usr/local/bin/emry

# The dashboard reads a directory of run logs from /logs. Mount a volume here
# (a PVC or hostPath in Kubernetes; see deploy/helm/emry). Owned by the emry
# user so file-mode writers running as the same user can populate it.
RUN mkdir -p /logs && chown emry:emry /logs
VOLUME ["/logs"]

USER emry
WORKDIR /home/emry

# The live web dashboard binds this port (default 8787 in `emry web`).
EXPOSE 8787

# tini reaps zombies and forwards signals to `emry`.
ENTRYPOINT ["/usr/bin/tini", "--", "emry"]

# Serve the multi-run project dashboard over /logs. NOTE: `emry web` takes the
# log directory via `--project <PATH>` (there is no separate --log-dir flag for
# the web subcommand); `--project` is the directory it scans for runs.
# --host 0.0.0.0 so the dashboard is reachable from outside the container; set
# EMRY_AUTH_TOKEN (and TLS) when exposing it beyond localhost.
CMD ["web", "--project", "/logs", "--port", "8787", "--host", "0.0.0.0"]
