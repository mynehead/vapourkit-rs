use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver};

use eframe::egui;
use serde::{Deserialize, Serialize};
use vk_core::chain::{self, FilterTemplate, Param, Step, StepKind};
use vk_core::encoding::{self, Codec, Encoding};
use vk_core::job::{self, Event, Job, Tools};
use vk_core::script::{self, Backend, ScriptConfig};
use vk_core::workflow::{self, Workflow};

fn main() -> eframe::Result {
    eframe::run_native(
        "Vapourkit",
        eframe::NativeOptions::default(),
        Box::new(|cc| {
            let mut app = App::default();
            if let Some(saved) = cc.storage.and_then(|s| eframe::get_value(s, eframe::APP_KEY)) {
                app.saved = saved;
            }
            Ok(Box::new(app))
        }),
    )
}

/// What survives a restart.
#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
struct Saved {
    steps: Vec<Step>,
    encoding: Encoding,
    presets: Vec<(String, Encoding)>,
}

struct App {
    saved: Saved,
    preset_name: String,
    tools: Tools,
    library: Vec<FilterTemplate>,
    /// The script typed by hand; runs instead of the generated one while set.
    edited_script: Option<String>,
    input: Option<PathBuf>,
    output: Option<PathBuf>,
    backend: Backend,
    num_streams: u32,
    running: Option<(Job, Receiver<Event>)>,
    progress: (u64, u64),
    status: String,
    log: Vec<String>,
}

impl Default for App {
    fn default() -> Self {
        App {
            saved: Saved::default(),
            preset_name: String::new(),
            tools: Tools::default(),
            library: chain::bundled_filters(),
            edited_script: None,
            input: None,
            output: None,
            backend: if cfg!(windows) { Backend::DirectMl } else { Backend::Ncnn },
            num_streams: 2,
            running: None,
            progress: (0, 0),
            status: String::new(),
            log: Vec::new(),
        }
    }
}

const TIP_VIDEO: &str = "The video to upscale. Its audio and subtitles are copied into the output unchanged.";
const TIP_OUTPUT: &str = "Where the result is written. The file extension picks the container (.mkv, .mp4 or .mov).";
const TIP_BACKEND: &str = "The engine that runs the AI models on your graphics card.\n\n\
    TensorRT: fastest, NVIDIA RTX only, needs a model built into an .engine file.\n\
    DirectML: any graphics card on Windows, runs .onnx models directly.\n\
    NCNN Vulkan: any graphics card with Vulkan, on Windows and Linux, runs .onnx models directly.";
const TIP_STREAMS: &str = "How many frames the model works on at the same time. More streams keep the graphics card busier \
    and can be faster, but need more video memory. 2 is a good start; use 1 if you run out of video memory.";
const TIP_FP16: &str = "The precision the model file was exported in. fp16 (half precision) is faster and needs less video memory; \
    fp32 is full precision. It must match the model; the file name usually says (for example ..._fp16.onnx), \
    and it is set from the name when you add the model.";
const TIP_CODEC: &str = "H.264: plays everywhere. H.265: about half the size of H.264 at the same quality. \
    AV1: the smallest files, slow on the CPU. ProRes: very large files, meant for further editing.";
const TIP_ENCODER: &str = "Software encodes on the CPU: the best quality for the file size, but the slowest. \
    NVIDIA NVENC, AMD AMF and Intel QSV use the video encoder on that graphics card: much faster, slightly larger files.";
const TIP_PRESET: &str = "Encoding speed against compression: slower presets make smaller files at the same quality.";
const TIP_QUALITY: &str = "Lower is better quality and a bigger file.";
const TIP_ADVANCED: &str = "Type the ffmpeg video arguments yourself (for example -c:v libx265 -crf 16) instead of using the controls.";
const TIP_WORKFLOW: &str = "A .vkworkflow file holds the whole filter chain and the encoding settings. \
    It is the same format the original Vapourkit uses.";

/// A Browse… button followed by the chosen path, cut to fit with the full
/// path on hover; true when the button was clicked.
fn browse_row(ui: &mut egui::Ui, path: Option<&std::path::Path>) -> bool {
    ui.horizontal(|ui| {
        let clicked = ui.button("Browse…").clicked();
        match path {
            Some(p) => {
                let full = p.display().to_string();
                ui.add(egui::Label::new(&full).truncate()).on_hover_text(full);
            }
            None => {
                ui.weak("(none)");
            }
        }
        clicked
    })
    .inner
}

impl App {
    fn poll(&mut self) {
        let Some((_, rx)) = &self.running else { return };
        let mut done = None;
        for ev in rx.try_iter() {
            match ev {
                Event::Log(l) => self.log.push(l),
                Event::Progress { frame, total } => self.progress = (frame, total),
                Event::Done(r) => done = Some(r),
            }
        }
        if let Some(r) = done {
            self.status = match r {
                Ok(()) => "Done".into(),
                Err(e) => {
                    self.log.push(e.clone());
                    format!("Failed: {}", e.lines().next().unwrap_or_default())
                }
            };
            self.running = None;
        }
    }

    fn script_config(&self, input: &PathBuf) -> ScriptConfig {
        ScriptConfig {
            input: input.clone(),
            steps: self.saved.steps.clone(),
            backend: self.backend,
            num_streams: self.num_streams,
        }
    }

    fn start(&mut self) {
        let (Some(input), Some(output)) = (&self.input, &self.output) else { return };
        let missing = self.saved.steps.iter().find_map(|s| match &s.kind {
            StepKind::Model { path, .. } if s.enabled && !path.is_file() => Some(s.name.clone()),
            _ => None,
        });
        if let Some(name) = missing {
            self.status = format!("Model file of step \"{name}\" not found; pick it again");
            return;
        }
        let script = match &self.edited_script {
            Some(s) => s.clone(),
            None => script::render(&self.script_config(input), &job::script_path()),
        };
        let (tx, rx) = channel();
        self.log.clear();
        self.progress = (0, 0);
        self.status = "Running".into();
        let job = job::start(self.tools.clone(), script, input.clone(), self.saved.encoding.args(), output.clone(), tx);
        self.running = Some((job, rx));
    }

    fn load_workflow(&mut self) {
        let Some(p) = rfd::FileDialog::new().add_filter("Workflow", &["vkworkflow"]).pick_file() else { return };
        match std::fs::read_to_string(&p).map_err(|e| e.to_string()).and_then(|t| workflow::load(&t)) {
            Ok((w, notes)) => {
                self.saved.steps = w.steps;
                if let Some(e) = w.encoding {
                    self.saved.encoding = e;
                }
                self.log.extend(notes);
                self.status = format!("Loaded workflow \"{}\"", w.name);
            }
            Err(e) => self.status = format!("Cannot load {}: {e}", p.display()),
        }
    }

    fn save_workflow(&mut self) {
        let Some(p) = rfd::FileDialog::new().add_filter("Workflow", &["vkworkflow"]).save_file() else { return };
        let w = Workflow {
            name: p.file_stem().unwrap_or_default().to_string_lossy().into_owned(),
            steps: self.saved.steps.clone(),
            encoding: Some(self.saved.encoding.clone()),
        };
        self.status = match std::fs::write(&p, workflow::save(&w)) {
            Ok(()) => format!("Saved {}", p.display()),
            Err(e) => format!("Cannot save {}: {e}", p.display()),
        };
    }

    fn import_filters(&mut self) {
        for p in rfd::FileDialog::new().add_filter("Filter", &["vkfilter"]).pick_files().unwrap_or_default() {
            match std::fs::read_to_string(&p).map_err(|e| e.to_string()).and_then(|t| FilterTemplate::parse(&t)) {
                Ok(t) => self.saved.steps.push(t.to_step()),
                Err(e) => self.status = format!("Cannot read {}: {e}", p.display()),
            }
        }
    }

    fn chain_ui(&mut self, ui: &mut egui::Ui) {
        // (index, -1 up / +1 down / 0 remove), applied after the loop.
        let mut action = None;
        let count = self.saved.steps.len();
        for (i, step) in self.saved.steps.iter_mut().enumerate() {
            let id = ui.make_persistent_id(("step", i));
            egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false)
                .show_header(ui, |ui| {
                    ui.checkbox(&mut step.enabled, "").on_hover_text("Run this step; untick to skip it without removing it");
                    ui.label(&step.name);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button("Remove").on_hover_text("Remove this step").clicked() {
                            action = Some((i, 0));
                        }
                        if ui.add_enabled(i + 1 < count, egui::Button::new("Down").small()).on_hover_text("Run this step later").clicked() {
                            action = Some((i, 1));
                        }
                        if ui.add_enabled(i > 0, egui::Button::new("Up").small()).on_hover_text("Run this step earlier").clicked() {
                            action = Some((i, -1));
                        }
                    });
                })
                .body(|ui| step_ui(ui, step));
        }
        match action {
            Some((i, 0)) => {
                self.saved.steps.remove(i);
            }
            Some((i, -1)) => self.saved.steps.swap(i, i - 1),
            Some((i, _)) => self.saved.steps.swap(i, i + 1),
            None => {}
        }
        if self.saved.steps.is_empty() {
            ui.weak("No steps: the video is only re-encoded.");
        }

        ui.horizontal_wrapped(|ui| {
            if ui.button("+ AI model…").on_hover_text("Add an upscaling or restoration model (.onnx, or .engine for TensorRT)").clicked() {
                if let Some(p) = rfd::FileDialog::new().add_filter("Model", &["onnx", "engine"]).pick_file() {
                    self.saved.steps.push(Step::model(p));
                }
            }
            ui.menu_button("+ Filter", |ui| {
                ui.label("Filters shipped with Vapourkit, by category. Some need extra VapourSynth plugins.");
                egui::ScrollArea::vertical().max_height(400.0).show(ui, |ui| {
                    for (category, group) in self.library.chunk_by(|a, b| a.category() == b.category()).map(|g| (g[0].category(), g)) {
                        ui.menu_button(category, |ui| {
                            for t in group {
                                if ui.button(&t.name).on_hover_text(&t.description).clicked() {
                                    self.saved.steps.push(t.to_step());
                                    ui.close_menu();
                                }
                            }
                        });
                    }
                });
            });
            if ui.button("Import .vkfilter…").on_hover_text("Add filters from .vkfilter files, Vapourkit's filter format").clicked() {
                self.import_filters();
            }
            ui.separator();
            if ui.button("Load workflow…").on_hover_text(TIP_WORKFLOW).clicked() {
                self.load_workflow();
            }
            if ui.button("Save workflow…").on_hover_text(TIP_WORKFLOW).clicked() {
                self.save_workflow();
            }
        });
    }

    fn script_ui(&mut self, ui: &mut egui::Ui) {
        match &mut self.edited_script {
            None => {
                ui.label("The script is generated from the steps above when you start.");
                let input = self.input.clone();
                if ui.add_enabled(input.is_some(), egui::Button::new("Edit the script by hand")).on_disabled_hover_text("Pick a video first").clicked() {
                    let cfg = self.script_config(input.as_ref().unwrap());
                    self.edited_script = Some(script::render(&cfg, &job::script_path()));
                }
            }
            Some(text) => {
                ui.colored_label(ui.visuals().warn_fg_color, "This script runs as typed; changes to the steps above are ignored until you discard it.");
                ui.add(egui::TextEdit::multiline(text).code_editor().desired_rows(20).desired_width(f32::INFINITY));
                if ui.button("Discard and use the generated script").clicked() {
                    self.edited_script = None;
                }
            }
        }
    }

    fn encoding_ui(&mut self, ui: &mut egui::Ui) {
        let e = &mut self.saved.encoding;
        // Switching to advanced starts from the simple settings' arguments.
        if ui.checkbox(&mut e.advanced, "Advanced: my own ffmpeg video arguments").on_hover_text(TIP_ADVANCED).changed() && e.advanced && e.custom_args.is_empty() {
            e.custom_args = Encoding { advanced: false, ..e.clone() }.args().join(" ");
        }
        if e.advanced {
            ui.add(egui::TextEdit::multiline(&mut e.custom_args).code_editor().desired_rows(2).desired_width(f32::INFINITY));
        } else {
            egui::Grid::new("encoding").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                ui.label("Codec").on_hover_text(TIP_CODEC);
                egui::ComboBox::from_id_salt("codec").selected_text(e.codec.label()).show_ui(ui, |ui| {
                    for c in Codec::ALL {
                        ui.selectable_value(&mut e.codec, c, c.label());
                    }
                }).response.on_hover_text(TIP_CODEC);
                ui.end_row();
                ui.label("Encoder").on_hover_text(TIP_ENCODER);
                egui::ComboBox::from_id_salt("encoder").selected_text(e.encoder.label()).show_ui(ui, |ui| {
                    for &enc in e.codec.encoders() {
                        ui.selectable_value(&mut e.encoder, enc, enc.label());
                    }
                }).response.on_hover_text(TIP_ENCODER);
                ui.end_row();
                e.normalize();
                if e.codec.has_quality() {
                    ui.label("Preset").on_hover_text(TIP_PRESET);
                    egui::ComboBox::from_id_salt("preset").selected_text(e.preset.as_str()).show_ui(ui, |ui| {
                        for &p in encoding::presets(e.codec, e.encoder) {
                            ui.selectable_value(&mut e.preset, p.to_string(), p);
                        }
                    }).response.on_hover_text(TIP_PRESET);
                    ui.end_row();
                    let (min, max, _) = encoding::quality_range(e.codec, e.encoder);
                    ui.label(e.encoder.quality_label()).on_hover_text(TIP_QUALITY);
                    ui.add(egui::Slider::new(&mut e.quality, min..=max)).on_hover_text(TIP_QUALITY);
                    ui.end_row();
                }
            });
        }

        ui.horizontal(|ui| {
            egui::ComboBox::from_id_salt("presets").selected_text("Load preset…").show_ui(ui, |ui| {
                if self.saved.presets.is_empty() {
                    ui.weak("No presets saved yet");
                }
                for (name, enc) in &self.saved.presets {
                    if ui.selectable_label(false, name).clicked() {
                        self.saved.encoding = enc.clone();
                        self.preset_name = name.clone();
                    }
                }
            });
            ui.add(egui::TextEdit::singleline(&mut self.preset_name).hint_text("Preset name").desired_width(140.0));
            let name = self.preset_name.trim().to_string();
            if ui.add_enabled(!name.is_empty(), egui::Button::new("Save")).on_hover_text("Save the encoding settings above under this name").clicked() {
                self.saved.presets.retain(|(n, _)| *n != name);
                self.saved.presets.push((name.clone(), self.saved.encoding.clone()));
            }
            let exists = self.saved.presets.iter().any(|(n, _)| *n == name);
            if ui.add_enabled(exists, egui::Button::new("Delete")).clicked() {
                self.saved.presets.retain(|(n, _)| *n != name);
            }
        });
    }
}

impl eframe::App for App {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, &self.saved);
    }

    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        self.poll();
        let busy = self.running.is_some();

        egui::TopBottomPanel::bottom("run").resizable(true).default_height(180.0).show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                let missing: Vec<&str> = [(self.input.is_none(), "a video"), (self.output.is_none(), "an output file")]
                    .into_iter()
                    .filter_map(|(m, what)| m.then_some(what))
                    .collect();
                let why = if busy { "A video is being processed".to_string() } else { format!("Pick {} first", missing.join(" and ")) };
                if ui.add_enabled(missing.is_empty() && !busy, egui::Button::new("Upscale Video"))
                    .on_disabled_hover_text(why)
                    .clicked()
                {
                    self.start();
                }
                if ui.add_enabled(busy, egui::Button::new("Cancel")).clicked() {
                    if let Some((job, _)) = &self.running {
                        job.cancel();
                    }
                }
                ui.label(&self.status);
            });
            let (frame, total) = self.progress;
            let fraction = if total > 0 { frame as f32 / total as f32 } else { 0.0 };
            ui.add(egui::ProgressBar::new(fraction).text(format!("{frame} / {total} frames")));
            ui.separator();
            egui::ScrollArea::vertical().stick_to_bottom(true).auto_shrink(false).show(ui, |ui| {
                for line in &self.log {
                    ui.monospace(line);
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.add_enabled_ui(!busy, |ui| {
                    egui::Grid::new("form").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                        ui.label("Video").on_hover_text(TIP_VIDEO);
                        if browse_row(ui, self.input.as_deref()) {
                            if let Some(p) = rfd::FileDialog::new().pick_file() {
                                self.output.get_or_insert_with(|| p.with_file_name(format!(
                                    "{}_upscaled.mkv",
                                    p.file_stem().unwrap_or_default().to_string_lossy()
                                )));
                                self.input = Some(p);
                            }
                        }
                        ui.end_row();

                        ui.label("Output").on_hover_text(TIP_OUTPUT);
                        if browse_row(ui, self.output.as_deref()) {
                            if let Some(p) = rfd::FileDialog::new().add_filter("Video", &["mkv", "mp4", "mov"]).save_file() {
                                self.output = Some(p);
                            }
                        }
                        ui.end_row();

                        ui.label("Backend").on_hover_text(TIP_BACKEND);
                        egui::ComboBox::from_id_salt("backend").selected_text(self.backend.label()).show_ui(ui, |ui| {
                            for b in Backend::ALL {
                                ui.selectable_value(&mut self.backend, b, b.label());
                            }
                        }).response.on_hover_text(TIP_BACKEND);
                        ui.end_row();

                        ui.label("Streams").on_hover_text(TIP_STREAMS);
                        ui.add(egui::DragValue::new(&mut self.num_streams).range(1..=8)).on_hover_text(TIP_STREAMS);
                        ui.end_row();
                    });
                    ui.separator();
                    egui::CollapsingHeader::new("Filter chain").default_open(true).show(ui, |ui| self.chain_ui(ui));
                    egui::CollapsingHeader::new("Encoding").default_open(true).show(ui, |ui| self.encoding_ui(ui));
                    egui::CollapsingHeader::new("Script").show(ui, |ui| self.script_ui(ui));
                });
            });
        });
        if busy {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
    }
}

/// One step's settings: model file and precision, or variables and code.
fn step_ui(ui: &mut egui::Ui, step: &mut Step) {
    ui.horizontal(|ui| {
        ui.label("Name");
        ui.text_edit_singleline(&mut step.name);
    });
    match &mut step.kind {
        StepKind::Model { path, fp16 } => {
            if browse_row(ui, Some(path)) {
                if let Some(p) = rfd::FileDialog::new().add_filter("Model", &["onnx", "engine"]).pick_file() {
                    *path = p;
                }
            }
            ui.checkbox(fp16, "fp16 model").on_hover_text(TIP_FP16);
        }
        StepKind::Custom { code, variables, parameters } => {
            egui::Grid::new(ui.id().with("vars")).num_columns(2).show(ui, |ui| {
                for (key, var) in variables.iter().filter(|(_, v)| !v.hidden) {
                    let Some(default) = var.default.clone() else { continue };
                    let whole = matches!(default, Param::Num(n) if n.fract() == 0.0);
                    let label = ui.label(key);
                    if let Some(d) = &var.description {
                        label.on_hover_text(d);
                    }
                    match parameters.entry(key.clone()).or_insert(default) {
                        Param::Num(n) if whole => {
                            ui.add(egui::DragValue::new(n).speed(1.0).fixed_decimals(0));
                        }
                        Param::Num(n) => {
                            ui.add(egui::DragValue::new(n).speed(0.01));
                        }
                        Param::Bool(b) => {
                            ui.checkbox(b, "");
                        }
                        Param::Str(s) => {
                            ui.text_edit_singleline(s);
                        }
                    }
                    ui.end_row();
                }
            });
            ui.add(egui::TextEdit::multiline(code).code_editor().desired_rows(4).desired_width(f32::INFINITY));
        }
    }
}
