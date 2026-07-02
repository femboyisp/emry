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

/// Access role a token grants. `Admin` outranks `Viewer` (derived `Ord` follows
/// declaration order), so an admin satisfies any viewer-level requirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Role {
    /// Read a single run's live dashboard.
    Viewer,
    /// Everything a viewer can, plus the multi-run project dashboard.
    Admin,
}

/// Opt-in dashboard security: bearer tokens (per role) and optional `tls`.
///
/// [`WebSecurity::default`] is fully open plain HTTP — the historical behavior.
/// `token` is the full-access (admin) token; `viewer_token` grants read-only
/// access to single-run dashboards but not the project overlay.
#[derive(Debug, Clone, Default)]
pub struct WebSecurity {
    /// Full-access bearer token (admin). `None` ⇒ no admin token.
    pub token: Option<String>,
    /// Read-only bearer token (viewer): single-run dashboards only.
    pub viewer_token: Option<String>,
    /// TLS cert/key to serve HTTPS. `None` ⇒ plain HTTP.
    pub tls: Option<TlsConfig>,
}

/// Middleware state: the configured tokens and the minimum role this server
/// requires. Held behind the layer so each request can resolve its role.
#[derive(Clone)]
struct AuthState {
    admin: Option<Arc<String>>,
    viewer: Option<Arc<String>>,
    min_role: Role,
}

impl AuthState {
    /// Resolves a presented token to the role it grants, if any (constant-time
    /// compared against each configured token).
    fn role_of(&self, presented: &str) -> Option<Role> {
        if let Some(admin) = &self.admin {
            if ct_eq(presented.as_bytes(), admin.as_bytes()) {
                return Some(Role::Admin);
            }
        }
        if let Some(viewer) = &self.viewer {
            if ct_eq(presented.as_bytes(), viewer.as_bytes()) {
                return Some(Role::Viewer);
            }
        }
        None
    }
}

impl WebSecurity {
    /// Wraps `router` with role-aware auth middleware requiring at least
    /// `min_role`. With no tokens configured the router is returned unchanged
    /// (fully open — the historical default).
    pub fn apply(&self, router: Router, min_role: Role) -> Router {
        if self.token.is_none() && self.viewer_token.is_none() {
            return router;
        }
        let state = AuthState {
            admin: self.token.clone().map(Arc::new),
            viewer: self.viewer_token.clone().map(Arc::new),
            min_role,
        };
        router.layer(axum::middleware::from_fn_with_state(state, require_role))
    }
}

/// Middleware: `/healthz` is always open; otherwise the presented token must map
/// to a role that meets the server's `min_role` (401 if unknown, 403 if too low).
async fn require_role(
    State(auth): State<AuthState>,
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    if req.uri().path() == "/healthz" {
        return Ok(next.run(req).await);
    }
    match presented_token(&req).and_then(|tok| auth.role_of(&tok)) {
        Some(role) if role >= auth.min_role => Ok(next.run(req).await),
        Some(_) => Err(StatusCode::FORBIDDEN),
        None => Err(StatusCode::UNAUTHORIZED),
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
    // The WebSocket upgrade can't set headers, so accept `?token=`. Parse with
    // form_urlencoded so the value is percent-decoded and `&` inside other
    // params can't confuse the split — a token with spaces/`%`/`+` still matches.
    req.uri().query().and_then(|q| {
        form_urlencoded::parse(q.as_bytes())
            .find(|(key, _)| key == "token")
            .map(|(_, value)| value.into_owned())
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

/// Serves `router` on `addr`, applying `security` (role-aware token layer
/// requiring at least `min_role` + TLS choice).
///
/// # Errors
///
/// Returns an [`std::io::Error`] if the address cannot be bound, the TLS
/// material cannot be loaded, or serving fails.
pub(crate) async fn serve_router(
    addr: SocketAddr,
    router: Router,
    security: WebSecurity,
    min_role: Role,
) -> std::io::Result<()> {
    let app = security.apply(router, min_role);
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

    fn routes() -> Router {
        use axum::routing::get;
        Router::new()
            .route("/", get(|| async { "ok" }))
            .route("/healthz", get(|| async { "ok" }))
    }

    /// Admin token `s3cret`, viewer token `peek`, requiring at least `min_role`.
    fn secured(min_role: Role) -> Router {
        WebSecurity {
            token: Some("s3cret".into()),
            viewer_token: Some("peek".into()),
            tls: None,
        }
        .apply(routes(), min_role)
    }

    fn guarded_router() -> Router {
        secured(Role::Viewer)
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
        // Even an admin-gated server is open when no tokens are configured.
        assert_eq!(
            status(
                WebSecurity::default().apply(routes(), Role::Admin),
                "/",
                None
            )
            .await,
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn rbac_gates_admin_routes_from_viewers() {
        // On an admin-required server (the project overlay):
        // admin token → 200, viewer token → 403, unknown → 401.
        assert_eq!(
            status(secured(Role::Admin), "/", Some("Bearer s3cret")).await,
            StatusCode::OK
        );
        assert_eq!(
            status(secured(Role::Admin), "/", Some("Bearer peek")).await,
            StatusCode::FORBIDDEN
        );
        assert_eq!(
            status(secured(Role::Admin), "/", Some("Bearer nope")).await,
            StatusCode::UNAUTHORIZED
        );
        // On a viewer-required server (single-run), the viewer token is enough,
        // and an admin (higher role) still passes.
        assert_eq!(
            status(secured(Role::Viewer), "/", Some("Bearer peek")).await,
            StatusCode::OK
        );
        assert_eq!(
            status(secured(Role::Viewer), "/", Some("Bearer s3cret")).await,
            StatusCode::OK
        );
        // /healthz stays open on the admin server too.
        assert_eq!(
            status(secured(Role::Admin), "/healthz", None).await,
            StatusCode::OK
        );
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

        // Query token is percent-decoded, so tokens with spaces/`+`/`%` match.
        let req = Request::builder()
            .uri("/ws?token=a%20b%2Bc")
            .body(Body::empty())
            .unwrap();
        assert_eq!(presented_token(&req).as_deref(), Some("a b+c"));

        // A `token`-suffixed param name (e.g. `xtoken`) is not mistaken for it.
        let req = Request::builder()
            .uri("/ws?xtoken=nope")
            .body(Body::empty())
            .unwrap();
        assert_eq!(presented_token(&req), None);

        // Non-bearer auth scheme is ignored.
        let req = Request::builder()
            .uri("/")
            .header(AUTHORIZATION, "Basic abc")
            .body(Body::empty())
            .unwrap();
        assert_eq!(presented_token(&req), None);
    }
}
