//! Optional transport security for the web dashboard: a bearer token and TLS.
//!
//! Both are opt-in and off by default, preserving the original local-only,
//! plain-HTTP posture. When a token is set, every route except `/healthz`
//! requires `Authorization: Bearer <token>` — or, for the WebSocket (which
//! cannot set request headers), a `?token=<token>` query parameter. `/healthz`
//! stays open so liveness/readiness probes work without credentials.
//!
//! When [`TlsConfig`] is set, the server is served over HTTPS from the given
//! PEM cert/key; Emry never generates certificates itself.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::{header::AUTHORIZATION, StatusCode};
use axum::middleware::Next;
use axum::response::Response;
use axum::Router;
use axum_server::tls_rustls::RustlsConfig;

/// PEM-encoded certificate chain and private key paths for serving HTTPS.
#[derive(Debug, Clone)]
pub struct TlsConfig {
    /// Path to the PEM certificate chain.
    pub cert: PathBuf,
    /// Path to the PEM private key.
    pub key: PathBuf,
}

/// Opt-in dashboard security: an optional bearer `token` and optional `tls`.
///
/// [`WebSecurity::default`] is fully open plain HTTP — the historical behavior.
#[derive(Debug, Clone, Default)]
pub struct WebSecurity {
    /// Bearer token required on every route except `/healthz`. `None` ⇒ open.
    pub token: Option<String>,
    /// TLS cert/key to serve HTTPS. `None` ⇒ plain HTTP.
    pub tls: Option<TlsConfig>,
}

impl WebSecurity {
    /// Wraps `router` with token-auth middleware when a token is configured;
    /// otherwise returns it unchanged.
    pub fn apply(&self, router: Router) -> Router {
        match &self.token {
            Some(token) => router.layer(axum::middleware::from_fn_with_state(
                Arc::new(token.clone()),
                require_token,
            )),
            None => router,
        }
    }
}

/// Middleware: allow `/healthz` unauthenticated; otherwise require a bearer
/// token (header or `?token=` query) matching `expected`.
async fn require_token(
    State(expected): State<Arc<String>>,
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let ok = req.uri().path() == "/healthz"
        || presented_token(&req).is_some_and(|tok| ct_eq(tok.as_bytes(), expected.as_bytes()));
    if ok {
        Ok(next.run(req).await)
    } else {
        Err(StatusCode::UNAUTHORIZED)
    }
}

/// Extracts the presented token from the `Authorization: Bearer` header, falling
/// back to a `token=` query parameter (the WebSocket upgrade can't set headers).
fn presented_token(req: &Request) -> Option<String> {
    if let Some(value) = req.headers().get(AUTHORIZATION) {
        if let Some(tok) = value.to_str().ok().and_then(|s| s.strip_prefix("Bearer ")) {
            return Some(tok.to_string());
        }
    }
    req.uri().query().and_then(|q| {
        q.split('&')
            .find_map(|kv| kv.strip_prefix("token="))
            .map(str::to_string)
    })
}

/// Constant-time byte-slice equality, so token checks don't leak length/prefix
/// via response timing.
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Serves `router` on `addr`, applying `security` (token layer + TLS choice).
///
/// # Errors
///
/// Returns an [`std::io::Error`] if the address cannot be bound, the TLS
/// material cannot be loaded, or serving fails.
pub(crate) async fn serve_router(
    addr: SocketAddr,
    router: Router,
    security: WebSecurity,
) -> std::io::Result<()> {
    let app = security.apply(router);
    if let Some(tls) = &security.tls {
        let config = RustlsConfig::from_pem_file(&tls.cert, &tls.key).await?;
        axum_server::bind_rustls(addr, config)
            .serve(app.into_make_service())
            .await
    } else {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, app).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ct_eq_matches_only_identical_slices() {
        assert!(ct_eq(b"secret", b"secret"));
        assert!(!ct_eq(b"secret", b"secreto")); // length mismatch
        assert!(!ct_eq(b"secret", b"secreu")); // same length, differs
        assert!(ct_eq(b"", b""));
    }

    fn guarded_router() -> Router {
        use axum::routing::get;
        WebSecurity {
            token: Some("s3cret".into()),
            tls: None,
        }
        .apply(
            Router::new()
                .route("/", get(|| async { "ok" }))
                .route("/healthz", get(|| async { "ok" })),
        )
    }

    async fn status(router: Router, uri: &str, auth: Option<&str>) -> StatusCode {
        use axum::body::Body;
        use tower::ServiceExt;
        let mut builder = Request::builder().uri(uri);
        if let Some(value) = auth {
            builder = builder.header(AUTHORIZATION, value);
        }
        router
            .oneshot(builder.body(Body::empty()).unwrap())
            .await
            .unwrap()
            .status()
    }

    #[tokio::test]
    async fn token_gate_allows_valid_and_rejects_missing_or_wrong() {
        // Correct bearer token → through.
        assert_eq!(
            status(guarded_router(), "/", Some("Bearer s3cret")).await,
            StatusCode::OK
        );
        // Missing token → 401.
        assert_eq!(
            status(guarded_router(), "/", None).await,
            StatusCode::UNAUTHORIZED
        );
        // Wrong token → 401.
        assert_eq!(
            status(guarded_router(), "/", Some("Bearer nope")).await,
            StatusCode::UNAUTHORIZED
        );
        // WebSocket-style `?token=` query is accepted.
        assert_eq!(
            status(guarded_router(), "/?token=s3cret", None).await,
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn healthz_is_open_without_a_token() {
        assert_eq!(
            status(guarded_router(), "/healthz", None).await,
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn no_token_configured_leaves_router_open() {
        assert_eq!(
            status(
                WebSecurity::default().apply(guarded_router_inner()),
                "/",
                None
            )
            .await,
            StatusCode::OK
        );
    }

    fn guarded_router_inner() -> Router {
        use axum::routing::get;
        Router::new().route("/", get(|| async { "ok" }))
    }

    #[test]
    fn presented_token_reads_bearer_header_then_query() {
        use axum::body::Body;
        // Bearer header wins.
        let req = Request::builder()
            .uri("/ws?token=fromquery")
            .header(AUTHORIZATION, "Bearer fromheader")
            .body(Body::empty())
            .unwrap();
        assert_eq!(presented_token(&req).as_deref(), Some("fromheader"));

        // Falls back to the query param (WebSocket case).
        let req = Request::builder()
            .uri("/ws?foo=1&token=fromquery")
            .body(Body::empty())
            .unwrap();
        assert_eq!(presented_token(&req).as_deref(), Some("fromquery"));

        // Non-bearer auth scheme is ignored.
        let req = Request::builder()
            .uri("/")
            .header(AUTHORIZATION, "Basic abc")
            .body(Body::empty())
            .unwrap();
        assert_eq!(presented_token(&req), None);
    }
}
