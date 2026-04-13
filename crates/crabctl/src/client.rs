use crate::{
    error::Error,
    types::{CreateKeyRequest, KeyResponse, KeySummary},
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

    #[allow(dead_code)]
    async fn post_empty(&self, path: &str) -> Result<reqwest::Response, Error> {
        let resp = self
            .client
            .post(self.url(path))
            .bearer_auth(&self.token)
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
}
