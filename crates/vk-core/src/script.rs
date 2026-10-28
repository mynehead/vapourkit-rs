//! The .vpy script for a run, rendered from the same template the Electron
//! app ships (assets/vapoursynth_template.vpy).

use std::path::{Path, PathBuf};

use crate::chain::{self, Step, StepKind};

pub const TEMPLATE: &str = include_str!("../../../assets/vapoursynth_template.vpy");
/// The helpers every script imports; written beside the script for each run.
pub const RUNTIME_PY: &str = include_str!("../../../assets/vk_runtime.py");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    TensorRt,
    DirectMl,
    Ncnn,
}

impl Backend {
    // shortcut: MIGraphX is left out until ONNX input inspection is ported (milestone 2).
    pub const ALL: [Backend; 3] = [Backend::TensorRt, Backend::DirectMl, Backend::Ncnn];

    pub fn label(self) -> &'static str {
        match self {
            Backend::TensorRt => "TensorRT (.engine)",
            Backend::DirectMl => "DirectML",
            Backend::Ncnn => "NCNN Vulkan",
        }
    }

    pub fn short(self) -> &'static str {
        match self {
            Backend::TensorRt => "TRT",
            Backend::DirectMl => "DML",
            Backend::Ncnn => "NCNN",
        }
    }

    fn id(self) -> &'static str {
        match self {
            Backend::TensorRt => "tensorrt",
            Backend::DirectMl => "directml",
            Backend::Ncnn => "ncnn",
        }
    }

    /// The model call, as in electron/providers/*.ts `modelCallCode`.
    fn model_call(self, model: &str, fp16: bool, streams: u32) -> String {
        let fp16 = if fp16 { "True" } else { "False" };
        match self {
            Backend::TensorRt => format!("clip = core.trt.Model(clip, engine_path={model}, num_streams={streams})\n"),
            Backend::DirectMl => format!(
                "clip = core.ort.Model(clip, network_path={model}, num_streams={streams}, provider=\"DML\", device_id=0, fp16={fp16}, verbosity=2)\n"
            ),
            Backend::Ncnn => format!(
                "clip = core.ncnn.Model(clip, network_path={model}, num_streams={streams}, device_id=0, fp16={fp16})\n"
            ),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ScriptConfig {
    pub input: PathBuf,
    /// Run in order; disabled steps are skipped.
    pub steps: Vec<Step>,
    pub backend: Backend,
    pub num_streams: u32,
}

/// A Python string literal for a path, with forward slashes as the app writes them.
fn py_path(p: &Path) -> String {
    let s = p.to_string_lossy().replace('\\', "/");
    format!("\"{}\"", s.replace('"', "\\\""))
}

/// The bare path for the template's own quoted placeholders.
fn template_path(p: &Path) -> String {
    p.to_string_lossy().replace('\\', "/").replace('"', "\\\"")
}

fn backend_helper(cfg: &ScriptConfig, work_dir: &Path) -> String {
    let attrs = "\"tensorrt\": \"TRT\", \"directml\": \"ORT_DML\", \"ncnn\": \"NCNN_VK\", \"migraphx\": \"MIGX\"";
    let dir = py_path(work_dir);
    let first_model = cfg.steps.iter().find_map(|s| match &s.kind {
        StepKind::Model { path, .. } => path.parent(),
        _ => None,
    });
    let models = py_path(first_model.unwrap_or(work_dir));
    format!(
        "VK_BACKEND = \"{id}\"\n\
         VK_BUILD_ENV = {{}}\n\
         import sys as _vk_sys\n\
         if {dir} not in _vk_sys.path:\n    _vk_sys.path.insert(0, {dir})\n\
         import vk_runtime\n\
         from vk_runtime import vk_backend, vk_timed_output\n\
         vk_runtime.configure(VK_BACKEND, {{{attrs}}}, VK_BUILD_ENV, models_path={models}, trtexec_path=\"trtexec\")\n",
        id = cfg.backend.id(),
    )
}

fn model_code(cfg: &ScriptConfig, model: &Path, fp16: bool) -> String {
    let rgb = if fp16 { "vs.RGBH" } else { "vs.RGBS" };
    format!(
        "\
         if clip.format.id != {rgb}:\n    clip = core.resize.Bilinear(clip, format={rgb}, matrix_in_s=\"709\", primaries_in_s=\"709\", transfer_in_s=\"709\")\n\
         clip = core.std.Expr(clip, expr=['x 0 max 1 min'])\n\
         {call}\
         clip = core.resize.Point(clip, format=vs.YUV444P16, matrix_s=\"709\", primaries_s=\"709\", transfer_s=\"709\")\n",
        call = cfg.backend.model_call(&py_path(model), fp16, cfg.num_streams),
    )
}

fn filters_code(cfg: &ScriptConfig) -> String {
    let mut code = String::new();
    for step in cfg.steps.iter().filter(|s| s.enabled) {
        // One line, so a name cannot end the comment early.
        code += &format!("# {}\n", step.name.replace(['\r', '\n'], " "));
        code += &match &step.kind {
            StepKind::Model { path, fp16 } => model_code(cfg, path, *fp16),
            StepKind::Custom { code, variables, parameters } => chain::render_custom(code, variables, parameters) + "\n",
        };
        code += "\n";
    }
    code
}

/// The template's source line, and what replaces it: BestSource when it is
/// installed, else L-SMASH or FFMS2.
const BS_SOURCE: &str = "clip = core.bs.VideoSource(source=source_path, cachemode=3, timecodes=source_timecodes)";
// Without BestSource no timecodes file is written, so vk_timed_output keeps a constant rate.
const ANY_SOURCE: &str = "\
if hasattr(core, \"bs\"):
    clip = core.bs.VideoSource(source=source_path, cachemode=3, timecodes=source_timecodes)
elif hasattr(core, \"lsmas\"):
    clip = core.lsmas.LWLibavSource(source_path)
elif hasattr(core, \"ffms2\"):
    clip = core.ffms2.Source(source_path)
else:
    raise vs.Error(\"No attribute with the name bs exists: no source filter (BestSource, L-SMASH or FFMS2) is installed\")";

/// The script text for `script_path`; vk_runtime.py is expected in its folder,
/// and the template's timecodes file is written beside it.
pub fn render(cfg: &ScriptConfig, script_path: &Path) -> String {
    let work_dir = script_path.parent().unwrap_or(Path::new("."));
    let timecodes = script_path.with_extension("timecodes.txt");
    TEMPLATE
        .replace(BS_SOURCE, ANY_SOURCE)
        .replace("{{INPUT_VIDEO}}", &template_path(&cfg.input))
        .replace("{{SOURCE_TIMECODES}}", &template_path(&timecodes))
        .replace("{{OVERWRITE_MATRIX}}", "False")
        .replace("{{MATRIX_709}}", "False")
        .replace("{{DEFAULT_MATRIX}}", "709")
        .replace("{{DEFAULT_PRIMARIES}}", "709")
        .replace("{{DEFAULT_TRANSFER}}", "709")
        .replace("{{OUTPUT_FORMAT}}", "vs.YUV420P8")
        .replace("{{BACKEND_HELPER}}", &backend_helper(cfg, work_dir))
        .replace("{{FILTERS}}", &filters_code(cfg))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_every_placeholder() {
        let cfg = ScriptConfig {
            input: r"C:\vids\in put.mkv".into(),
            steps: vec![
                Step::model("/models/2x_x_fp16.onnx".into()),
                Step { enabled: false, ..chain::bundled_filters().into_iter().find(|t| t.name == "CAS Sharpen").unwrap().to_step() },
                chain::bundled_filters().into_iter().find(|t| t.name == "Crop").unwrap().to_step(),
            ],
            backend: Backend::Ncnn,
            num_streams: 2,
        };
        let s = render(&cfg, Path::new("/tmp/vk/run.vpy"));
        assert!(!s.contains("{{"), "unfilled placeholder:\n{s}");
        assert!(s.contains("source_path = \"C:/vids/in put.mkv\""));
        assert!(s.contains("/tmp/vk/run.timecodes.txt"));
        assert!(s.contains("core.ncnn.Model(clip, network_path=\"/models/2x_x_fp16.onnx\", num_streams=2, device_id=0, fp16=True)"));
        assert!(s.contains("format=vs.RGBH"));
        assert!(!s.contains("core.cas.CAS"), "disabled step was rendered");
        assert!(s.contains("left   = 0\n"));
        assert!(s.contains("core.lsmas.LWLibavSource(source_path)"), "source fallback missing");
    }
}
