use crate::{
    error::Error,
    types::{
        AuditRecord, BudgetEntry, CreateKeyRequest, CreateProviderRequest, KeyResponse, KeySummary,
        ProviderSummary, UsageEntry,
    },
};
use reqwest::StatusCode;
use serde::Deserialize;

/// Percent-encode a URL path segment (RFC 3986 unreserved chars pass through).
fn encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => {
                out.push_str(&format!("%{b:02X}"));
            }
        }
    }
    out
}

pub struct AdminClient {
    client: reqwest::Client,
    base_url: String,
    token: String,
}

/// Server error envelope: `{ "error": { "message": "..." } }`.
#[derive(Deserialize)]
struct ApiErrorEnvelope {
    error: ApiErrorBody,
}

#[derive(Deserialize)]
struct ApiErrorBody {
    message: String,
}

impl AdminClient {
    pub fn new(base_url: String, token: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            token,
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base_url)
    }

    async fn check(response: reqwest::Response) -> Result<reqwest::Response, Error> {
        if response.status().is_success() {
            return Ok(response);
        }
        let status = response.status().as_u16();
        let body = response.text().await.unwrap_or_default();
        let message = serde_json::from_str::<ApiErrorEnvelope>(&body)
            .map(|env| env.error.message)
            .unwrap_or_else(|_| {
                if body.is_empty() {
                    format!("request failed with status {status}")
                } else {
                    body
                }
            });
        Err(Error::Api { status, message })
    }

    async fn get(&self, path: &str) -> Result<reqwest::Response, Error> {
        let resp = self
            .client
            .get(self.url(path))
            .bearer_auth(&self.token)
            .send()
            .await?;
        Self::check(resp).await
    }

    /// GET a list endpoint, treating 404 as an empty list. Extension
    /// endpoints (usage, budget, audit logs) are only mounted when the
    /// extension is enabled; the CLI should render "no rows" in that
    /// case, not surface a 404 error to the user.
    async fn get_list<T: for<'de> Deserialize<'de>>(&self, path: &str) -> Result<Vec<T>, Error> {
        let resp = self
            .client
            .get(self.url(path))
            .bearer_auth(&self.token)
            .send()
            .await?;
        if resp.status() == StatusCode::NOT_FOUND {
            return Ok(Vec::new());
        }
        Ok(Self::check(resp).await?.json().await?)
    }

    async fn post_json<T: serde::Serialize>(
        &self,
        path: &str,
        body: &T,
    ) -> Result<reqwest::Response, Error> {
        let resp = self
            .client
            .post(self.url(path))
            .bearer_auth(&self.token)
            .json(body)
            .send()
            .await?;
        Self::check(resp).await
    }

    async fn patch_json<T: serde::Serialize>(
        &self,
        path: &str,
        body: &T,
    ) -> Result<reqwest::Response, Error> {
        let resp = self
            .client
            .patch(self.url(path))
            .bearer_auth(&self.token)
            .json(body)
            .send()
            .await?;
        Self::check(resp).await
    }

    async fn delete(&self, path: &str) -> Result<(), Error> {
        let resp = self
            .client
            .delete(self.url(path))
            .bearer_auth(&self.token)
            .send()
            .await?;
        // 204 No Content is success with no body.
        if resp.status() == StatusCode::NO_CONTENT {
            return Ok(());
        }
        Self::check(resp).await.map(|_| ())
    }

    // ── Key management ──

    pub async fn list_keys(&self) -> Result<Vec<KeySummary>, Error> {
        Ok(self.get("/v1/admin/keys").await?.json().await?)
    }

    pub async fn create_key(&self, req: &CreateKeyRequest) -> Result<KeyResponse, Error> {
        Ok(self.post_json("/v1/admin/keys", req).await?.json().await?)
    }

    pub async fn get_key(&self, name: &str) -> Result<KeySummary, Error> {
        let path = format!("/v1/admin/keys/{}", encode(name));
        Ok(self.get(&path).await?.json().await?)
    }

    pub async fn update_key(
        &self,
        name: &str,
        patch: &serde_json::Value,
    ) -> Result<KeySummary, Error> {
        let path = format!("/v1/admin/keys/{}", encode(name));
        Ok(self.patch_json(&path, patch).await?.json().await?)
    }

    pub async fn delete_key(&self, name: &str) -> Result<(), Error> {
        let path = format!("/v1/admin/keys/{}", encode(name));
        self.delete(&path).await
    }

    // ── Providers ──

    pub async fn list_providers(&self) -> Result<Vec<ProviderSummary>, Error> {
        Ok(self.get("/v1/admin/providers").await?.json().await?)
    }

    pub async fn get_provider(&self, name: &str) -> Result<ProviderSummary, Error> {
        let path = format!("/v1/admin/providers/{}", encode(name));
        Ok(self.get(&path).await?.json().await?)
    }

    pub async fn create_provider(
        &self,
        req: &CreateProviderRequest,
    ) -> Result<ProviderSummary, Error> {
        Ok(self
            .post_json("/v1/admin/providers", req)
            .await?
            .json()
            .await?)
    }

    pub async fn update_provider(
        &self,
        name: &str,
        patch: &serde_json::Value,
    ) -> Result<ProviderSummary, Error> {
        let path = format!("/v1/admin/providers/{}", encode(name));
        Ok(self.patch_json(&path, patch).await?.json().await?)
    }

    pub async fn delete_provider(&self, name: &str) -> Result<(), Error> {
        let path = format!("/v1/admin/providers/{}", encode(name));
        self.delete(&path).await
    }

    // ── Usage ──

    pub async fn usage(
        &self,
        name: Option<&str>,
        model: Option<&str>,
    ) -> Result<Vec<UsageEntry>, Error> {
        let mut path = String::from("/v1/admin/usage");
        append_query(&mut path, &[("name", name), ("model", model)]);
        self.get_list(&path).await
    }

    // ── Budget ──

    pub async fn budget(&self) -> Result<Vec<BudgetEntry>, Error> {
        self.get_list("/v1/budget").await
    }

    // ── Logs ──

    pub async fn logs(
        &self,
        key: Option<&str>,
        model: Option<&str>,
        since: Option<i64>,
        until: Option<i64>,
        limit: usize,
    ) -> Result<Vec<AuditRecord>, Error> {
        let since = since.map(|v| v.to_string());
        let until = until.map(|v| v.to_string());
        let limit = limit.to_string();
        let mut path = String::from("/v1/admin/logs");
        append_query(
            &mut path,
            &[
                ("key", key),
                ("model", model),
                ("since", since.as_deref()),
                ("until", until.as_deref()),
                ("limit", Some(&limit)),
            ],
        );
        self.get_list(&path).await
    }

    // ── Cache ──

    pub async fn clear_cache(&self) -> Result<(), Error> {
        self.delete("/v1/cache").await
    }
}

fn append_query(path: &mut String, params: &[(&str, Option<&str>)]) {
    let mut first = true;
    for (k, v) in params {
        if let Some(val) = v {
            path.push(if first { '?' } else { '&' });
            path.push_str(k);
            path.push('=');
            path.push_str(&encode(val));
            first = false;
        }
    }
}
