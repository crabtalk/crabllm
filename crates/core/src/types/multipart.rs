use bytes::Bytes;

/// A buffered multipart form field. Carries name, filename, content type, and
/// bytes across the provider trait boundary without depending on any HTTP
/// client crate.
#[derive(Debug, Clone)]
pub struct MultipartField {
    pub name: String,
    pub filename: Option<String>,
    pub content_type: Option<String>,
    pub bytes: Bytes,
}
