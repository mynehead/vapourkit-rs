//! .vkworkflow files: a filter chain plus encoding settings, in upstream's
//! TOML layout (electron/workflowHandlers.ts), so files move between the apps.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::chain::{Param, Step, StepKind, Variable};
use crate::encoding::{Codec, Encoder, Encoding};

pub struct Workflow {
    pub name: String,
    pub steps: Vec<Step>,
    pub encoding: Option<Encoding>,
}

#[derive(Serialize, Deserialize)]
struct File {
    workflow: Header,
    #[serde(default)]
    filters: Vec<Filter>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    encoding_settings: Option<EncodingSettings>,
}

#[derive(Serialize, Deserialize)]
struct Header {
    name: String,
    #[serde(default)]
    version: String,
    #[serde(default)]
    created_at: String,
    #[serde(default)]
    description: String,
}

fn yes() -> bool {
    true
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Filter {
    name: String,
    #[serde(default)]
    code: String,
    #[serde(default = "yes")]
    enabled: bool,
    #[serde(default)]
    order: i64,
    #[serde(default)]
    filter_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model_type: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    parameters: BTreeMap<String, Param>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    variables: BTreeMap<String, Variable>,
}

#[derive(Serialize, Deserialize)]
struct EncodingSettings {
    /// The resolved arguments, the only encoding field older upstream versions read.
    #[serde(default)]
    ffmpeg_args: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    encoding_mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    custom_args: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    simple_encoding: Option<Simple>,
}

#[derive(Serialize, Deserialize)]
struct Simple {
    codec: Codec,
    encoder: Encoder,
    preset: String,
    quality: u32,
}

/// Reads a workflow. Steps of kinds not ported yet (Load Video side chains)
/// are left out and named in the returned notes.
pub fn load(text: &str) -> Result<(Workflow, Vec<String>), String> {
    let mut file: File = toml::from_str(text).map_err(|e| e.to_string())?;
    file.filters.sort_by_key(|f| f.order);
    let mut notes = Vec::new();
    let mut steps = Vec::new();
    for f in file.filters {
        let kind = match f.filter_type.as_str() {
            // shortcut: upstream stores a portable model name, not a path; those steps need their model picked again until a models folder exists.
            "aiModel" => StepKind::Model {
                fp16: !f.model_path.as_deref().unwrap_or("").to_lowercase().contains("fp32"),
                path: f.model_path.unwrap_or_default().into(),
            },
            "custom" | "" => StepKind::Custom { code: f.code, variables: f.variables, parameters: f.parameters },
            other => {
                notes.push(format!("Skipped step \"{}\": {other} steps are not supported yet", f.name));
                continue;
            }
        };
        steps.push(Step { name: f.name, enabled: f.enabled, kind });
    }
    let encoding = file.encoding_settings.map(|e| {
        // Older files carry only ffmpeg_args, which then become the advanced arguments.
        let old_style = e.simple_encoding.is_none() && !e.ffmpeg_args.is_empty();
        let mut enc = Encoding {
            advanced: e.encoding_mode.as_deref() == Some("advanced") || old_style,
            custom_args: e.custom_args.unwrap_or(e.ffmpeg_args),
            ..Encoding::default()
        };
        if let Some(s) = e.simple_encoding {
            (enc.codec, enc.encoder, enc.preset, enc.quality) = (s.codec, s.encoder, s.preset, s.quality);
            enc.normalize();
        }
        enc
    });
    Ok((Workflow { name: file.workflow.name, steps, encoding }, notes))
}

pub fn save(w: &Workflow) -> String {
    let filters = w
        .steps
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let mut f = Filter {
                name: s.name.clone(),
                code: String::new(),
                enabled: s.enabled,
                order: i as i64,
                filter_type: String::new(),
                model_path: None,
                model_type: None,
                parameters: BTreeMap::new(),
                variables: BTreeMap::new(),
            };
            match &s.kind {
                StepKind::Model { path, .. } => {
                    f.filter_type = "aiModel".into();
                    f.model_path = Some(path.to_string_lossy().into_owned());
                    f.model_type = Some("image".into());
                }
                StepKind::Custom { code, variables, parameters } => {
                    f.filter_type = "custom".into();
                    f.code = code.clone();
                    f.variables = variables.clone();
                    f.parameters = parameters.clone();
                }
            }
            f
        })
        .collect();
    let encoding_settings = w.encoding.as_ref().map(|e| EncodingSettings {
        ffmpeg_args: e.args().join(" "),
        encoding_mode: Some(if e.advanced { "advanced" } else { "simple" }.into()),
        custom_args: Some(e.custom_args.clone()),
        simple_encoding: Some(Simple { codec: e.codec, encoder: e.encoder, preset: e.preset.clone(), quality: e.quality }),
    });
    let file = File {
        workflow: Header { name: w.name.clone(), version: "1.0".into(), created_at: String::new(), description: String::new() },
        filters,
        encoding_settings,
    };
    toml::to_string(&file).expect("workflow serializes")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::bundled_filters;

    #[test]
    fn round_trips() {
        let crop = bundled_filters().into_iter().find(|t| t.name == "Crop").unwrap().to_step();
        let mut enc = Encoding { codec: Codec::H265, encoder: Encoder::Nvidia, ..Default::default() };
        enc.normalize();
        let w = Workflow {
            name: "Test".into(),
            steps: vec![Step::model("/m/2x_fp16.onnx".into()), Step { enabled: false, ..crop }],
            encoding: Some(enc.clone()),
        };
        let text = save(&w);
        let (back, notes) = load(&text).unwrap();
        assert!(notes.is_empty());
        assert_eq!(back.name, "Test");
        assert_eq!(back.steps, w.steps);
        assert_eq!(back.encoding, Some(enc));
    }

    #[test]
    fn reads_upstream_layout() {
        let text = r#"
[workflow]
name = "Upstream"
version = "1.0"
created_at = "2026-01-01T00:00:00.000Z"

[[filters]]
name = "Second"
code = "clip = core.std.Invert(clip)"
enabled = true
order = 1
filterType = "custom"

[[filters]]
name = "First"
code = "x = {{n}}"
enabled = true
order = 0
filterType = "custom"
parameters = { n = 3 }
[filters.variables.n]
type = "number"
default = 1

[[filters]]
name = "Load Video"
code = ""
enabled = true
order = 2
filterType = "loadVideo"

[encoding_settings]
ffmpeg_args = "-c:v libx264 -crf 16"
"#;
        let (w, notes) = load(text).unwrap();
        assert_eq!(w.steps.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(), ["First", "Second"]);
        assert_eq!(notes.len(), 1);
        let enc = w.encoding.unwrap();
        assert!(enc.advanced);
        assert_eq!(enc.args(), ["-c:v", "libx264", "-crf", "16"]);
    }
}
