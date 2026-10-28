mod theme;

use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver};
use std::time::Instant;

use eframe::egui::{self, RichText};
use serde::{Deserialize, Serialize};
use theme::{ACCENT, WEAK};
use vk_core::chain::{self, FilterTemplate, Param, Step, StepKind};
use vk_core::encoding::{self, Codec, Encoding};
use vk_core::job::{self, Event, Job, Tools};
use vk_core::probe::{self, VideoInfo};
use vk_core::script::{self, Backend, ScriptConfig};
use vk_core::setup;
use vk_core::workflow::{self, Workflow};

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1400.0, 860.0]).with_min_inner_size([900.0, 560.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Vapourkit",
        options,
        Box::new(|cc| {
            theme::apply(&cc.egui_ctx);
            let mut app = App::default();
            app.show_setup = setup::installed_vspipe().is_none();
            if let Some(saved) = cc.storage.and_then(|s| eframe::get_value(s, eframe::APP_KEY)) {
                app.saved = saved;
            }
            // `vapourkit some.mkv flow.vkworkflow`, as "Open with" passes files.
            for p in std::env::args_os().skip(1) {
                app.open_path(p.into());
            }
            Ok(Box::new(app))
        }),
    )
}

const TIP_SETUP: &str = "Install VapourSynth and the plugins Vapourkit needs, or add the plugin for another backend.";
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

/// What survives a restart.
#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
struct Saved {
    steps: Vec<Step>,
    encoding: Encoding,
    presets: Vec<(String, Encoding)>,
}

/// A source being probed in the background: its info and a still.
type Loading = Receiver<(Result<VideoInfo, String>, Option<Vec<u8>>)>;

struct App {
    saved: Saved,
    preset_name: String,
    tools: Tools,
    library: Vec<FilterTemplate>,
    /// The script typed by hand; runs instead of the generated one while set.
    edited_script: Option<String>,
    input: Option<PathBuf>,
    info: Option<VideoInfo>,
    loading: Option<Loading>,
    output: Option<PathBuf>,
    backend: Backend,
    num_streams: u32,
    running: Option<(Job, Receiver<Event>)>,
    /// When the first frame came out, and its number; the speed is measured
    /// from there so script loading and indexing do not count.
    started: Option<(Instant, u64)>,
    progress: (u64, u64),
    /// The source still, replaced by the output's frames while a run goes.
    preview: Option<egui::TextureHandle>,
    status: String,
    log: Vec<String>,
    show_console: bool,
    show_setup: bool,
    /// A setup run: its output lines and, at the end, the vspipe it installed.
    setup: Option<(Receiver<String>, Receiver<Result<PathBuf, String>>)>,
    setup_status: String,
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
            info: None,
            loading: None,
            output: None,
            backend: if cfg!(windows) { Backend::DirectMl } else { Backend::Ncnn },
            num_streams: 2,
            running: None,
            started: None,
            progress: (0, 0),
            preview: None,
            status: String::new(),
            log: Vec::new(),
            show_console: false,
            show_setup: false,
            setup: None,
            setup_status: String::new(),
        }
    }
}

/// A Browse… button followed by the chosen path, cut to fit with the full
/// path on hover; true when the button was clicked.
fn browse_row(ui: &mut egui::Ui, path: Option<&Path>) -> bool {
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

fn timecode(secs: f64) -> String {
    let cs = (secs * 100.0).round() as u64;
    format!("{:02}:{:02}:{:02}.{:02}", cs / 360_000, cs / 6000 % 60, cs / 100 % 60, cs % 100)
}

fn decode_jpeg(jpeg: &[u8]) -> Option<egui::ColorImage> {
    let img = image::load_from_memory(jpeg).ok()?.to_rgba8();
    Some(egui::ColorImage::from_rgba_unmultiplied([img.width() as usize, img.height() as usize], img.as_raw()))
}

impl App {
    fn set_preview(&mut self, ctx: &egui::Context, jpeg: &[u8]) {
        let Some(img) = decode_jpeg(jpeg) else { return };
        match &mut self.preview {
            Some(t) => t.set(img, egui::TextureOptions::LINEAR),
            None => self.preview = Some(ctx.load_texture("preview", img, egui::TextureOptions::LINEAR)),
        }
    }

    fn set_input(&mut self, p: PathBuf) {
        self.output = Some(p.with_file_name(format!("{}_upscaled.mkv", p.file_stem().unwrap_or_default().to_string_lossy())));
        self.info = None;
        self.preview = None;
        let (tx, rx) = channel();
        let tools = self.tools.clone();
        let path = p.clone();
        std::thread::spawn(move || {
            let info = probe::probe(&tools.ffprobe, &path);
            let at = info.as_ref().map_or(0.0, |i| i.duration * 0.25);
            let still = probe::still(&tools.ffmpeg, &path, at).ok();
            let _ = tx.send((info, still));
        });
        self.loading = Some(rx);
        self.input = Some(p);
    }

    fn poll(&mut self, ctx: &egui::Context) {
        if let Some(rx) = &self.loading {
            if let Ok((info, still)) = rx.try_recv() {
                self.loading = None;
                match info {
                    Ok(i) => self.info = Some(i),
                    Err(e) => self.log.push(format!("ffprobe: {e}")),
                }
                if let Some(jpeg) = still {
                    self.set_preview(ctx, &jpeg);
                }
            }
        }
        self.poll_setup();
        let Some((_, rx)) = &self.running else { return };
        let events: Vec<Event> = rx.try_iter().collect();
        for ev in events {
            match ev {
                Event::Log(l) => self.log.push(l),
                Event::Progress { frame, total } => {
                    self.progress = (frame, total);
                    if frame > 0 {
                        self.started.get_or_insert((Instant::now(), frame));
                    }
                }
                Event::Preview(jpeg) => self.set_preview(ctx, &jpeg),
                Event::Done(r) => {
                    self.status = match r {
                        Ok(()) => "Done".into(),
                        Err(e) => {
                            self.log.push(e.clone());
                            self.show_console = true;
                            self.show_setup |= e.starts_with(job::MISSING_PLUGIN);
                            format!("Failed: {}", e.lines().next().unwrap_or_default())
                        }
                    };
                    self.running = None;
                    self.started = None;
                    return;
                }
            }
        }
    }

    fn script_config(&self, input: &Path) -> ScriptConfig {
        ScriptConfig {
            input: input.to_path_buf(),
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
        self.status = "Starting…".into();
        self.started = None;
        let job = job::start(self.tools.clone(), script, input.clone(), self.saved.encoding.args(), output.clone(), tx);
        self.running = Some((job, rx));
    }

    fn load_workflow_file(&mut self, p: &Path) {
        match std::fs::read_to_string(p).map_err(|e| e.to_string()).and_then(|t| workflow::load(&t)) {
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

    fn import_filter_file(&mut self, p: &Path) {
        match std::fs::read_to_string(p).map_err(|e| e.to_string()).and_then(|t| FilterTemplate::parse(&t)) {
            Ok(t) => self.saved.steps.push(t.to_step()),
            Err(e) => self.status = format!("Cannot read {}: {e}", p.display()),
        }
    }

    /// A file dropped on the window or passed on the command line: a
    /// workflow, a filter, a model, or else the video.
    fn open_path(&mut self, p: PathBuf) {
        match p.extension().and_then(|e| e.to_str()).map(str::to_lowercase).as_deref() {
            Some("vkworkflow") => self.load_workflow_file(&p),
            Some("vkfilter") => self.import_filter_file(&p),
            Some("onnx" | "engine") => self.saved.steps.push(Step::model(p)),
            _ if self.running.is_none() => self.set_input(p),
            _ => {}
        }
    }

    fn handle_drops(&mut self, ctx: &egui::Context) {
        let dropped: Vec<PathBuf> = ctx.input(|i| i.raw.dropped_files.iter().filter_map(|f| f.path.clone()).collect());
        for p in dropped {
            self.open_path(p);
        }
    }

    fn top_bar(&mut self, ui: &mut egui::Ui, busy: bool) {
        ui.horizontal(|ui| {
            ui.label(RichText::new("✨").color(ACCENT).size(18.0));
            ui.label(RichText::new("VAPOURKIT").strong().size(15.0).extra_letter_spacing(2.5));
            ui.separator();
            ui.label(RichText::new("WORKFLOW").small().color(WEAK).extra_letter_spacing(1.5));
            ui.add_enabled_ui(!busy, |ui| {
                if ui.button("🗁 Open").on_hover_text(TIP_WORKFLOW).clicked() {
                    if let Some(p) = rfd::FileDialog::new().add_filter("Workflow", &["vkworkflow"]).pick_file() {
                        self.load_workflow_file(&p);
                    }
                }
                if ui.button("💾 Save").on_hover_text(TIP_WORKFLOW).clicked() {
                    self.save_workflow();
                }
                if ui.button("⬇ Import filter").on_hover_text("Add filters from .vkfilter files, Vapourkit's filter format").clicked() {
                    for p in rfd::FileDialog::new().add_filter("Filter", &["vkfilter"]).pick_files().unwrap_or_default() {
                        self.import_filter_file(&p);
                    }
                }
                ui.separator();
                egui::ComboBox::from_id_salt("backend")
                    .selected_text(RichText::new(format!("⚙ {}", self.backend.short())).strong())
                    .show_ui(ui, |ui| {
                        for b in Backend::ALL {
                            ui.selectable_value(&mut self.backend, b, b.label());
                        }
                    })
                    .response
                    .on_hover_text(TIP_BACKEND);
                ui.label(RichText::new("STREAMS").small().color(WEAK)).on_hover_text(TIP_STREAMS);
                ui.add(egui::DragValue::new(&mut self.num_streams).range(1..=8)).on_hover_text(TIP_STREAMS);
                ui.separator();
                if ui.selectable_label(self.show_setup, "⬇ Setup").on_hover_text(TIP_SETUP).clicked() {
                    self.show_setup = !self.show_setup;
                }
            });
        });
    }

    fn status_bar(&mut self, ui: &mut egui::Ui, busy: bool) {
        let (frame, total) = self.progress;
        let fraction = if total > 0 { frame as f32 / total as f32 } else { 0.0 };
        if busy {
            ui.add(egui::ProgressBar::new(fraction).desired_height(3.0).fill(ACCENT));
        }
        ui.horizontal(|ui| {
            if busy {
                let fps = self.started.map_or(0.0, |(at, first)| {
                    let secs = at.elapsed().as_secs_f64();
                    if secs > 0.5 { (frame - first) as f64 / secs } else { 0.0 }
                });
                theme::card().show(ui, |ui| {
                    ui.label(format!("Processing frame {frame}/{total}"));
                });
                ui.label(RichText::new(format!("{:.0}%", fraction * 100.0)).strong());
                if fps > 0.0 {
                    ui.label(RichText::new("Speed").color(WEAK));
                    ui.label(RichText::new(format!("{fps:.1} FPS")).strong());
                }
                if fps > 0.0 && total > frame {
                    ui.label(RichText::new("ETA").color(WEAK));
                    ui.label(RichText::new(timecode((total - frame) as f64 / fps)).strong());
                }
            } else {
                ui.label(&self.status);
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let missing: Vec<&str> = [(self.input.is_none(), "a video"), (self.output.is_none(), "an output file")]
                    .into_iter()
                    .filter_map(|(m, what)| m.then_some(what))
                    .collect();
                let why = if busy {
                    "A video is being processed".to_string()
                } else if self.setup.is_some() {
                    "Setup is still running".to_string()
                } else {
                    format!("Pick {} first", missing.join(" and "))
                };
                if ui.add_enabled(missing.is_empty() && !busy && self.setup.is_none(), theme::primary("▶ Start processing")).on_disabled_hover_text(why).clicked() {
                    self.start();
                }
                if ui.selectable_label(self.show_console, ">_ Console").clicked() {
                    self.show_console = !self.show_console;
                }
                if busy && ui.add(theme::danger("⏹ Stop")).clicked() {
                    if let Some((job, _)) = &self.running {
                        job.cancel();
                    }
                }
            });
        });
    }

    fn preview_ui(&mut self, ui: &mut egui::Ui) {
        theme::section(ui, "PREVIEW", if self.running.is_some() { "output, 1 frame per second" } else { "" }, |_| {});
        let rect = ui.available_rect_before_wrap();
        ui.painter().rect_filled(rect, 0.0, theme::BG);
        match &self.preview {
            Some(tex) => {
                let size = tex.size_vec2();
                let scale = (rect.width() / size.x).min(rect.height() / size.y);
                let r = egui::Rect::from_center_size(rect.center(), size * scale);
                ui.painter().image(tex.id(), r, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
            }
            None => {
                let text = if self.loading.is_some() { "Reading the video…" } else { "Drop a video here, or choose one under Source" };
                ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, text, egui::FontId::proportional(16.0), WEAK);
            }
        }
        ui.allocate_rect(rect, egui::Sense::hover());
    }

    fn source_ui(&mut self, ui: &mut egui::Ui) {
        theme::section(ui, "SOURCE", "", |_| {});
        theme::card().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                if ui.add(egui::Button::new(RichText::new("🎞").size(18.0)).min_size(egui::vec2(44.0, 36.0))).on_hover_text(TIP_VIDEO).clicked() {
                    if let Some(p) = rfd::FileDialog::new().pick_file() {
                        self.set_input(p);
                    }
                }
                ui.vertical(|ui| match &self.input {
                    Some(p) => {
                        let name = p.file_name().unwrap_or_default().to_string_lossy().into_owned();
                        ui.add(egui::Label::new(RichText::new(name).strong()).truncate()).on_hover_text(p.display().to_string());
                        let line = match &self.info {
                            Some(i) => format!(
                                "{}x{} · {:.3} fps · {:.2} MB · {}",
                                i.width,
                                i.height,
                                i.fps,
                                i.size_bytes as f64 / 1_048_576.0,
                                timecode(i.duration)
                            )
                            .replace(".000 fps", " fps"),
                            None => "…".into(),
                        };
                        ui.label(RichText::new(line).monospace().color(WEAK));
                    }
                    None => {
                        ui.label(RichText::new("No video chosen").strong());
                        ui.label(RichText::new("Click the button or drop a file onto the window").color(WEAK));
                    }
                });
            });
        });
    }

    fn filters_ui(&mut self, ui: &mut egui::Ui) {
        let on = self.saved.steps.iter().filter(|s| s.enabled).count();
        let note = format!("{on}/{} on", self.saved.steps.len());
        let mut add_model = false;
        let mut add: Option<Step> = None;
        let library = &self.library;
        theme::section(ui, "FILTERS", &note, |ui| {
            ui.menu_button(RichText::new("✚ Add filter").color(ACCENT), |ui| {
                ui.set_max_width(260.0);
                if ui.button("✨ AI model…").on_hover_text("An upscaling or restoration model (.onnx, or .engine for TensorRT)").clicked() {
                    add_model = true;
                    ui.close_menu();
                }
                ui.separator();
                ui.label(RichText::new("Shipped filters. Some need extra VapourSynth plugins.").small().color(WEAK));
                egui::ScrollArea::vertical().max_height(420.0).show(ui, |ui| {
                    for group in library.chunk_by(|a, b| a.category() == b.category()) {
                        ui.menu_button(group[0].category(), |ui| {
                            for t in group {
                                if ui.button(&t.name).on_hover_text(&t.description).clicked() {
                                    add = Some(t.to_step());
                                    ui.close_menu();
                                }
                            }
                        });
                    }
                });
            });
        });
        if add_model {
            if let Some(p) = rfd::FileDialog::new().add_filter("Model", &["onnx", "engine"]).pick_file() {
                add = Some(Step::model(p));
            }
        }
        self.saved.steps.extend(add);

        enum Act {
            Up,
            Down,
            Copy,
            Remove,
        }
        // Applied after the loop, which borrows the steps.
        let mut action = None;
        let count = self.saved.steps.len();
        for (i, step) in self.saved.steps.iter_mut().enumerate() {
            let id = ui.make_persistent_id(("step", i));
            let mut state = egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false);
            let is_model = matches!(step.kind, StepKind::Model { .. });
            let frame = if is_model && step.enabled { theme::card().stroke(egui::Stroke::new(1.0_f32, theme::ACCENT_DIM)) } else { theme::card() };
            frame.show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    let number = RichText::new(format!(" {} ", i + 1)).monospace().background_color(theme::BG);
                    ui.label(number.color(if step.enabled { ACCENT } else { WEAK }));
                    let icon = if is_model { "✨ " } else { "" };
                    let name = RichText::new(format!("{icon}{}", step.name)).strong().color(if step.enabled { theme::TEXT } else { WEAK });
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button("✖").on_hover_text("Remove this step").clicked() {
                            action = Some((i, Act::Remove));
                        }
                        if ui.small_button("🗐").on_hover_text("Duplicate this step").clicked() {
                            action = Some((i, Act::Copy));
                        }
                        ui.checkbox(&mut step.enabled, "").on_hover_text("Run this step; untick to skip it without removing it");
                        if ui.add_enabled(i + 1 < count, egui::Button::new("⏷").small()).on_hover_text("Run this step later").clicked() {
                            action = Some((i, Act::Down));
                        }
                        if ui.add_enabled(i > 0, egui::Button::new("⏶").small()).on_hover_text("Run this step earlier").clicked() {
                            action = Some((i, Act::Up));
                        }
                        ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                            let header = ui.add(egui::Label::new(name).truncate().sense(egui::Sense::click()));
                            if header.on_hover_text("Click to show or hide the settings").clicked() {
                                state.toggle(ui);
                            }
                        });
                    });
                });
                state.show_body_unindented(ui, |ui| step_ui(ui, step));
            });
        }
        match action {
            Some((i, Act::Remove)) => {
                self.saved.steps.remove(i);
            }
            Some((i, Act::Copy)) => {
                let copy = self.saved.steps[i].clone();
                self.saved.steps.insert(i + 1, copy);
            }
            Some((i, Act::Up)) => self.saved.steps.swap(i, i - 1),
            Some((i, Act::Down)) => self.saved.steps.swap(i, i + 1),
            None => {}
        }
        if self.saved.steps.is_empty() {
            theme::card().show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.label(RichText::new("No steps yet. Add an AI model or a filter; with none, the video is only re-encoded.").color(WEAK));
            });
        }
    }

    fn output_ui(&mut self, ui: &mut egui::Ui) {
        let note = self.output.as_ref().and_then(|p| p.extension()).map(|e| e.to_string_lossy().to_uppercase()).unwrap_or_default();
        theme::section(ui, "OUTPUT", &note, |_| {});
        theme::card().show(ui, |ui| {
            ui.set_width(ui.available_width());
            theme::field_label(ui, "Save to").on_hover_text(TIP_OUTPUT);
            if browse_row(ui, self.output.as_deref()) {
                if let Some(p) = rfd::FileDialog::new().add_filter("Video", &["mkv", "mp4", "mov"]).save_file() {
                    self.output = Some(p);
                }
            }
            ui.add_space(4.0);
            self.encoding_ui(ui);
        });
    }

    fn encoding_ui(&mut self, ui: &mut egui::Ui) {
        let e = &mut self.saved.encoding;
        ui.horizontal(|ui| {
            theme::field_label(ui, "Video codec").on_hover_text(TIP_CODEC);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                // Switching to advanced starts from the simple settings' arguments.
                if ui.toggle_value(&mut e.advanced, "☰ Advanced").on_hover_text(TIP_ADVANCED).changed() && e.advanced && e.custom_args.is_empty() {
                    e.custom_args = Encoding { advanced: false, ..e.clone() }.args().join(" ");
                }
            });
        });
        if e.advanced {
            ui.add(egui::TextEdit::multiline(&mut e.custom_args).code_editor().desired_rows(2).desired_width(f32::INFINITY));
        } else {
            let width = ui.available_width();
            egui::ComboBox::from_id_salt("codec")
                .width(width)
                .selected_text(e.codec.label())
                .show_ui(ui, |ui| {
                    for c in Codec::ALL {
                        ui.selectable_value(&mut e.codec, c, c.label());
                    }
                })
                .response
                .on_hover_text(TIP_CODEC);
            theme::field_label(ui, "Encoder").on_hover_text(TIP_ENCODER);
            egui::ComboBox::from_id_salt("encoder")
                .width(width)
                .selected_text(e.encoder.label())
                .show_ui(ui, |ui| {
                    for &enc in e.codec.encoders() {
                        ui.selectable_value(&mut e.encoder, enc, enc.label());
                    }
                })
                .response
                .on_hover_text(TIP_ENCODER);
            e.normalize();
            if e.codec.has_quality() {
                ui.columns(2, |cols| {
                    theme::field_label(&mut cols[0], "Preset").on_hover_text(TIP_PRESET);
                    let w = cols[0].available_width();
                    egui::ComboBox::from_id_salt("preset")
                        .width(w)
                        .selected_text(e.preset.as_str())
                        .show_ui(&mut cols[0], |ui| {
                            for &p in encoding::presets(e.codec, e.encoder) {
                                ui.selectable_value(&mut e.preset, p.to_string(), p);
                            }
                        })
                        .response
                        .on_hover_text(TIP_PRESET);
                    theme::field_label(&mut cols[1], e.encoder.quality_label()).on_hover_text(TIP_QUALITY);
                    let (min, max, _) = encoding::quality_range(e.codec, e.encoder);
                    cols[1].add(egui::Slider::new(&mut e.quality, min..=max)).on_hover_text(TIP_QUALITY);
                });
            }
        }

        theme::field_label(ui, "Presets");
        ui.horizontal(|ui| {
            egui::ComboBox::from_id_salt("presets").selected_text("Load…").show_ui(ui, |ui| {
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
            ui.add(egui::TextEdit::singleline(&mut self.preset_name).hint_text("Preset name").desired_width(130.0));
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

    fn script_ui(&mut self, ui: &mut egui::Ui) {
        let note = if self.edited_script.is_some() { "edited by hand" } else { "generated" };
        theme::section(ui, "SCRIPT", note, |_| {});
        theme::card().show(ui, |ui| {
            ui.set_width(ui.available_width());
            match &mut self.edited_script {
                None => {
                    ui.label(RichText::new("The VapourSynth script is generated from the filters above when you start.").color(WEAK));
                    let input = self.input.clone();
                    if ui.add_enabled(input.is_some(), egui::Button::new("✏ Edit the script by hand")).on_disabled_hover_text("Choose a video first").clicked() {
                        let cfg = self.script_config(input.as_deref().unwrap());
                        self.edited_script = Some(script::render(&cfg, &job::script_path()));
                    }
                }
                Some(text) => {
                    ui.colored_label(ui.visuals().warn_fg_color, "This script runs as typed; changes to the filters above are ignored until you discard it.");
                    ui.add(egui::TextEdit::multiline(text).code_editor().desired_rows(18).desired_width(f32::INFINITY));
                    if ui.button("Discard and use the generated script").clicked() {
                        self.edited_script = None;
                    }
                }
            }
        });
    }

    fn poll_setup(&mut self) {
        let Some((lines, done)) = &self.setup else { return };
        for l in lines.try_iter() {
            self.setup_status = l.clone();
            self.log.push(l);
        }
        if let Ok(r) = done.try_recv() {
            self.setup_status = match r {
                Ok(vspipe) => {
                    self.tools.vspipe = vspipe;
                    "Done. Vapourkit now uses this VapourSynth.".into()
                }
                Err(e) => {
                    self.log.push(e.clone());
                    format!("Failed: {e}")
                }
            };
            self.setup = None;
        }
    }

    fn setup_window(&mut self, ctx: &egui::Context) {
        let mut open = self.show_setup;
        egui::Window::new("Setup").open(&mut open).collapsible(false).resizable(false).default_width(480.0).show(ctx, |ui| {
            ui.label(format!(
                "Vapourkit runs on VapourSynth. Setup installs a private copy for Vapourkit alone: \
                 {}, VapourSynth R79 with the plugin for the backend below, and everything the shipped filters need: \
                 the VapourSynth plugins and scripts, PyTorch and the RIFE and DPIR models.",
                if cfg!(windows) { "Python 3.13" } else { "a Python environment" }
            ));
            ui.label(RichText::new(format!("Into {}", setup::data_dir().display())).small().color(WEAK));
            ui.label(RichText::new("It downloads about 2 GB from python.org, PyPI and GitHub, and takes a while; with TensorRT \
                (CUDA PyTorch and TensorRT) several GB more. Run it again to add the plugin for another backend.").small().color(WEAK));
            ui.add_space(4.0);
            let running = self.setup.is_some();
            ui.horizontal(|ui| {
                theme::field_label(ui, "Backend");
                ui.add_enabled_ui(!running, |ui| {
                    egui::ComboBox::from_id_salt("setup_backend").selected_text(self.backend.label()).show_ui(ui, |ui| {
                        for b in Backend::ALL {
                            ui.selectable_value(&mut self.backend, b, b.label());
                        }
                    });
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let label = if setup::installed_vspipe().is_some() { "Install again" } else { "Install" };
                    if ui.add_enabled(!running && self.running.is_none(), theme::primary(label)).clicked() {
                        let (line_tx, line_rx) = channel();
                        let (done_tx, done_rx) = channel();
                        let backend = self.backend;
                        std::thread::spawn(move || {
                            let _ = done_tx.send(setup::run(backend, &line_tx));
                        });
                        self.setup = Some((line_rx, done_rx));
                    }
                });
            });
            if running {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.add(egui::Label::new(&self.setup_status).truncate());
                });
            } else if !self.setup_status.is_empty() {
                ui.add(egui::Label::new(&self.setup_status).wrap());
            } else if let Some(p) = setup::installed_vspipe() {
                ui.label(RichText::new(format!("Installed: {}", p.display())).small().color(ACCENT));
            }
        });
        self.show_setup = open;
    }
}

impl eframe::App for App {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, &self.saved);
    }

    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        self.poll(ctx);
        self.handle_drops(ctx);
        let busy = self.running.is_some();
        let bar = egui::Frame::new().fill(theme::BG).inner_margin(egui::Margin::symmetric(12, 8));

        egui::TopBottomPanel::top("top").frame(bar).show(ctx, |ui| self.top_bar(ui, busy));
        egui::TopBottomPanel::bottom("status").frame(bar).show(ctx, |ui| self.status_bar(ui, busy));
        egui::SidePanel::right("side").resizable(true).default_width(460.0).min_width(380.0).show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.add_space(6.0);
                ui.add_enabled_ui(!busy, |ui| {
                    self.source_ui(ui);
                    ui.add_space(6.0);
                    self.filters_ui(ui);
                    ui.add_space(6.0);
                    self.output_ui(ui);
                    ui.add_space(6.0);
                    self.script_ui(ui);
                });
                ui.add_space(6.0);
            });
        });
        egui::CentralPanel::default().frame(egui::Frame::new().fill(theme::PANEL).inner_margin(12)).show(ctx, |ui| self.preview_ui(ui));

        let mut open = self.show_console;
        egui::Window::new("Console").open(&mut open).default_size([700.0, 320.0]).show(ctx, |ui| {
            if ui.button("Copy all").clicked() {
                ctx.copy_text(self.log.join("\n"));
            }
            egui::ScrollArea::vertical().stick_to_bottom(true).auto_shrink(false).show(ui, |ui| {
                for line in &self.log {
                    ui.monospace(line);
                }
            });
        });
        self.show_console = open;
        self.setup_window(ctx);

        if busy || self.loading.is_some() || self.setup.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
    }
}

/// One step's settings: model file and precision, or variables and code.
fn step_ui(ui: &mut egui::Ui, step: &mut Step) {
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        theme::field_label(ui, "Name");
        ui.add(egui::TextEdit::singleline(&mut step.name).desired_width(f32::INFINITY));
    });
    match &mut step.kind {
        StepKind::Model { path, fp16 } => {
            theme::field_label(ui, "Model file");
            if browse_row(ui, Some(path)) {
                if let Some(p) = rfd::FileDialog::new().add_filter("Model", &["onnx", "engine"]).pick_file() {
                    *path = p;
                }
            }
            ui.checkbox(fp16, "fp16 model").on_hover_text(TIP_FP16);
        }
        StepKind::Custom { code, variables, parameters } => {
            if variables.values().any(|v| !v.hidden && v.default.is_some()) {
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
            }
            ui.add(egui::TextEdit::multiline(code).code_editor().desired_rows(4).desired_width(f32::INFINITY));
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn timecode_format() {
        assert_eq!(super::timecode(9.16), "00:00:09.16");
        assert_eq!(super::timecode(3725.5), "01:02:05.50");
    }
}
