use std::fmt;

pub enum Error {
    Http(reqwest::Error),
    Api { status: u16, message: String },
    Config(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Http(e) => write!(f, "http error: {e}"),
            Self::Api { status, message } => write!(f, "API error ({status}): {message}"),
            Self::Config(msg) => write!(f, "config error: {msg}"),
        }
    }
}

impl From<reqwest::Error> for Error {
    fn from(e: reqwest::Error) -> Self {
        Self::Http(e)
    }
}
