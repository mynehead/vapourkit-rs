//! The filter chain: AI model steps and custom VapourSynth code steps, and the
//! .vkfilter templates custom steps start from (upstream's format, unchanged).

use std::collections::BTreeMap;
use std::path::PathBuf;

use include_dir::{include_dir, Dir};
use serde::{Deserialize, Serialize};

static BUNDLED_FILTERS: Dir = include_dir!("$CARGO_MANIFEST_DIR/../../assets/filters");

/// A value for a `{{variable}}` placeholder.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Param {
    Bool(bool),
    Num(f64),
    Str(String),
}

impl Param {
    /// The Python literal, as upstream's renderCustomFilterCode writes it.
    fn to_python(&self) -> String {
        match self {
            Param::Bool(b) => if *b { "True" } else { "False" }.into(),
            Param::Num(n) => n.to_string(),
            Param::Str(s) => format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")),
        }
    }
}

/// A variable a .vkfilter exposes, from its `[variables.<name>]` table.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Variable {
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<Param>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The app's to fill, not shown as a control.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub hidden: bool,
}

/// A .vkfilter file.
#[derive(Clone, Debug, Deserialize)]
pub struct FilterTemplate {
    pub name: String,
    #[serde(default)]
    pub code: String,
    #[serde(default)]
    pub description: String,
    /// A string or a list of strings upstream; the first one is used here.
    #[serde(default)]
    category: Option<toml::Value>,
    #[serde(default)]
    pub variables: BTreeMap<String, Variable>,
}

impl FilterTemplate {
    pub fn parse(text: &str) -> Result<Self, String> {
        toml::from_str(text).map_err(|e| e.to_string())
    }

    pub fn category(&self) -> &str {
        match &self.category {
            Some(toml::Value::String(s)) => s,
            Some(toml::Value::Array(a)) => a.first().and_then(|v| v.as_str()).unwrap_or("Other"),
            _ => "Other",
        }
    }

    pub fn to_step(&self) -> Step {
        Step {
            name: self.name.clone(),
            enabled: true,
            kind: StepKind::Custom {
                code: self.code.clone(),
                variables: self.variables.clone(),
                parameters: BTreeMap::new(),
            },
        }
    }
}

/// The filters shipped with the app, sorted by category then name.
pub fn bundled_filters() -> Vec<FilterTemplate> {
    let mut out: Vec<FilterTemplate> = BUNDLED_FILTERS
        .files()
        .filter_map(|f| FilterTemplate::parse(f.contents_utf8()?).ok())
        // shortcut: steps reading another step's picture ({{stage:x}}) need the stage graph; hidden until it is ported.
        .filter(|t| !t.code.contains("{{stage"))
        .collect();
    out.sort_by(|a, b| (a.category(), &a.name).cmp(&(b.category(), &b.name)));
    out
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum StepKind {
    Model {
        path: PathBuf,
        /// fp16 models take RGBH, fp32 models RGBS.
        fp16: bool,
    },
    Custom {
        code: String,
        variables: BTreeMap<String, Variable>,
        parameters: BTreeMap<String, Param>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Step {
    pub name: String,
    pub enabled: bool,
    pub kind: StepKind,
}

impl Step {
    pub fn model(path: PathBuf) -> Step {
        let fp16 = !path.to_string_lossy().to_lowercase().contains("fp32");
        Step {
            name: path.file_stem().unwrap_or_default().to_string_lossy().into_owned(),
            enabled: true,
            kind: StepKind::Model { path, fp16 },
        }
    }
}

/// Custom code with its declared `{{variables}}` filled; undeclared ones and
/// ordinary Python braces are left alone.
pub fn render_custom(code: &str, variables: &BTreeMap<String, Variable>, parameters: &BTreeMap<String, Param>) -> String {
    let code = code.trim();
    let mut out = String::with_capacity(code.len());
    let mut rest = code;
    while let Some(start) = rest.find("{{") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        let value = after.find("}}").and_then(|end| {
            let key = after[..end].trim();
            let value = parameters.get(key).or(variables.get(key)?.default.as_ref())?;
            Some((end, value.to_python()))
        });
        match value {
            Some((end, py)) => {
                out.push_str(&py);
                rest = &after[end + 2..];
            }
            None => {
                out.push_str("{{");
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fills_declared_variables_only() {
        let t = FilterTemplate::parse(
            r#"
name = "T"
code = """
a = {{ strength }}
b = {{name}}
c = {{undeclared}}
d = {"k": {{flag}}}"""
category = ["Sharpening"]
[variables.strength]
type = "number"
default = 1.0
[variables.name]
type = "string"
default = 'say "hi"'
[variables.flag]
type = "boolean"
default = true
"#,
        )
        .unwrap();
        assert_eq!(t.category(), "Sharpening");
        let mut params = BTreeMap::new();
        params.insert("strength".to_string(), Param::Num(0.5));
        let out = render_custom(&t.code, &t.variables, &params);
        assert_eq!(out, "a = 0.5\nb = \"say \\\"hi\\\"\"\nc = {{undeclared}}\nd = {\"k\": True}");
        assert_eq!(render_custom("x = {{strength}}", &t.variables, &BTreeMap::new()), "x = 1");
    }

    #[test]
    fn bundled_filters_parse() {
        let all = bundled_filters();
        assert!(all.len() > 100, "only {} filters parsed", all.len());
        assert!(all.iter().any(|t| t.name == "CAS Sharpen"));
    }
}
