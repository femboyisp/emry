//! Multi-run project dashboard: overlay every run in a log directory.
//!
//! Unlike the live single-run dashboard ([`crate::server`]), this serves a
//! **static snapshot** of a whole directory of runs — each run's per-metric
//! series, loaded once from `metrics.jsonl` — so a sweep can be compared on one
//! chart. Reload the page to pick up new/updated runs.
//!
//! `emry-web` stays free of `emry-store`: the caller (the CLI) loads the runs
//! and passes them in as these serializable types.

use axum::response::{Html, IntoResponse, Json};
use axum::routing::get;
use axum::{extract::State, Router};
use serde::Serialize;
use std::net::SocketAddr;
use std::sync::Arc;

/// One metric's series within a run.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ProjectSeries {
    /// Metric name.
    pub label: String,
    /// Step of each value (parallel to `values`).
    pub steps: Vec<u64>,
    /// Values (parallel to `steps`).
    pub values: Vec<f64>,
}

/// One run in the project (its directory name + its metric series).
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ProjectRun {
    /// Run-directory name (`{project}_{timestamp}`).
    pub name: String,
    /// The run's metric series, in first-seen order.
    pub metrics: Vec<ProjectSeries>,
}

/// A snapshot of all runs under a log directory, served to the overlay chart.
#[derive(Debug, Clone, Serialize, Default, PartialEq)]
pub struct Project {
    /// Runs, newest first.
    pub runs: Vec<ProjectRun>,
}

const PROJECT_HTML: &str = include_str!("project.html");

/// Builds the project-dashboard router over an immutable [`Project`] snapshot.
pub fn app_project(project: Arc<Project>) -> Router {
    Router::new()
        .route("/", get(|| async { Html(PROJECT_HTML) }))
        .route("/healthz", get(|| async { "ok" }))
        .route("/project", get(project_json))
        .with_state(project)
}

async fn project_json(State(project): State<Arc<Project>>) -> impl IntoResponse {
    Json((*project).clone())
}

/// Binds `addr` and serves the project dashboard (a static snapshot of `project`).
///
/// # Errors
///
/// Returns an [`std::io::Error`] if the address cannot be bound or serving fails.
pub async fn serve_project(addr: SocketAddr, project: Project) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app_project(Arc::new(project))).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    fn sample() -> Project {
        Project {
            runs: vec![ProjectRun {
                name: "llama_20260101_120000".into(),
                metrics: vec![ProjectSeries {
                    label: "loss".into(),
                    steps: vec![0, 1],
                    values: vec![1.0, 0.5],
                }],
            }],
        }
    }

    #[tokio::test]
    async fn project_route_serves_the_snapshot() {
        let resp = app_project(Arc::new(sample()))
            .oneshot(
                Request::builder()
                    .uri("/project")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let json = std::str::from_utf8(&body).unwrap();
        assert!(json.contains("llama_20260101_120000") && json.contains("\"loss\""));
    }

    #[tokio::test]
    async fn index_serves_self_hosted_html() {
        let resp = app_project(Arc::new(Project::default()))
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let html = std::str::from_utf8(&body).unwrap();
        assert!(html.contains("<canvas"));
        assert!(html.contains("/project"));
        assert!(!html.contains("https://"), "no CDN — air-gap friendly");
    }

    #[tokio::test]
    async fn healthz_ok() {
        let resp = app_project(Arc::new(Project::default()))
            .oneshot(
                Request::builder()
                    .uri("/healthz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }
}
