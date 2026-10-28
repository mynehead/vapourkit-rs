use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver};

use eframe::egui;
use serde::{Deserialize, Serialize};
use vk_core::encoding::{self, Codec, Encoding};
use vk_core::job::{self, Event, Job, Tools};
use vk_core::script::{Backend, ScriptConfig};

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
    encoding: Encoding,
    presets: Vec<(String, Encoding)>,
}

struct App {
    saved: Saved,
    preset_name: String,
    tools: Tools,
    input: Option<PathBuf>,
    model: Option<PathBuf>,
    output: Option<PathBuf>,
    backend: Backend,
    fp16: bool,
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
            input: None,
            model: None,
            output: None,
            backend: if cfg!(windows) { Backend::DirectMl } else { Backend::Ncnn },
            fp16: true,
            num_streams: 2,
            running: None,
            progress: (0, 0),
            status: String::new(),
            log: Vec::new(),
        }
    }
}

fn path_label(p: &Option<PathBuf>) -> String {
    p.as_ref().map_or("(none)".into(), |p| p.display().to_string())
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

    fn start(&mut self) {
        let (Some(input), Some(model), Some(output)) = (&self.input, &self.model, &self.output) else { return };
        let cfg = ScriptConfig {
            input: input.clone(),
            model: model.clone(),
            backend: self.backend,
            fp16: self.fp16,
            num_streams: self.num_streams,
        };
        let (tx, rx) = channel();
        self.log.clear();
        self.progress = (0, 0);
        self.status = "Running".into();
        self.running = Some((job::start(self.tools.clone(), cfg, self.saved.encoding.args(), output.clone(), tx), rx));
    }

    fn encoding_ui(&mut self, ui: &mut egui::Ui) {
        let e = &mut self.saved.encoding;
        // Switching to advanced starts from the simple settings' arguments.
        if ui.checkbox(&mut e.advanced, "Advanced: my own ffmpeg video arguments").changed() && e.advanced && e.custom_args.is_empty() {
            e.custom_args = Encoding { advanced: false, ..e.clone() }.args().join(" ");
        }
        if e.advanced {
            ui.add(egui::TextEdit::multiline(&mut e.custom_args).code_editor().desired_rows(2).desired_width(f32::INFINITY));
        } else {
            egui::Grid::new("encoding").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                ui.label("Codec");
                egui::ComboBox::from_id_salt("codec").selected_text(e.codec.label()).show_ui(ui, |ui| {
                    for c in Codec::ALL {
                        ui.selectable_value(&mut e.codec, c, c.label());
                    }
                });
                ui.end_row();
                ui.label("Encoder");
                egui::ComboBox::from_id_salt("encoder").selected_text(e.encoder.label()).show_ui(ui, |ui| {
                    for &enc in e.codec.encoders() {
                        ui.selectable_value(&mut e.encoder, enc, enc.label());
                    }
                });
                ui.end_row();
                e.normalize();
                if e.codec.has_quality() {
                    ui.label("Preset");
                    egui::ComboBox::from_id_salt("preset").selected_text(e.preset.as_str()).show_ui(ui, |ui| {
                        for &p in encoding::presets(e.codec, e.encoder) {
                            ui.selectable_value(&mut e.preset, p.to_string(), p);
                        }
                    });
                    ui.end_row();
                    let (min, max, _) = encoding::quality_range(e.codec, e.encoder);
                    ui.label(e.encoder.quality_label()).on_hover_text("Lower is better quality and a bigger file");
                    ui.add(egui::Slider::new(&mut e.quality, min..=max));
                    ui.end_row();
                }
            });
        }

        ui.horizontal(|ui| {
            egui::ComboBox::from_id_salt("presets").selected_text("Load preset…").show_ui(ui, |ui| {
                for (name, enc) in &self.saved.presets {
                    if ui.selectable_label(false, name).clicked() {
                        self.saved.encoding = enc.clone();
                        self.preset_name = name.clone();
                    }
                }
            });
            ui.add(egui::TextEdit::singleline(&mut self.preset_name).hint_text("Preset name").desired_width(140.0));
            let name = self.preset_name.trim().to_string();
            if ui.add_enabled(!name.is_empty(), egui::Button::new("Save")).clicked() {
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
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_enabled_ui(!busy, |ui| {
                egui::Grid::new("form").num_columns(3).spacing([8.0, 6.0]).show(ui, |ui| {
                    ui.label("Video");
                    ui.label(path_label(&self.input));
                    if ui.button("Browse…").clicked() {
                        if let Some(p) = rfd::FileDialog::new().pick_file() {
                            self.output.get_or_insert_with(|| p.with_file_name(format!(
                                "{}_upscaled.mkv",
                                p.file_stem().unwrap_or_default().to_string_lossy()
                            )));
                            self.input = Some(p);
                        }
                    }
                    ui.end_row();

                    ui.label("Model");
                    ui.label(path_label(&self.model));
                    if ui.button("Browse…").clicked() {
                        if let Some(p) = rfd::FileDialog::new().add_filter("Model", &["onnx", "engine"]).pick_file() {
                            self.fp16 = !p.to_string_lossy().to_lowercase().contains("fp32");
                            self.model = Some(p);
                        }
                    }
                    ui.end_row();

                    ui.label("Output");
                    ui.label(path_label(&self.output));
                    if ui.button("Browse…").clicked() {
                        if let Some(p) = rfd::FileDialog::new().add_filter("Video", &["mkv", "mp4", "mov"]).save_file() {
                            self.output = Some(p);
                        }
                    }
                    ui.end_row();

                    ui.label("Backend");
                    egui::ComboBox::from_id_salt("backend").selected_text(self.backend.label()).show_ui(ui, |ui| {
                        for b in Backend::ALL {
                            ui.selectable_value(&mut self.backend, b, b.label());
                        }
                    });
                    ui.end_row();

                    ui.label("Precision");
                    ui.checkbox(&mut self.fp16, "fp16");
                    ui.end_row();

                    ui.label("Streams");
                    ui.add(egui::DragValue::new(&mut self.num_streams).range(1..=8));
                    ui.end_row();
                });
                ui.separator();
                egui::CollapsingHeader::new("Encoding").default_open(true).show(ui, |ui| self.encoding_ui(ui));
            });

            ui.separator();
            ui.horizontal(|ui| {
                let ready = self.input.is_some() && self.model.is_some() && self.output.is_some();
                if ui.add_enabled(ready && !busy, egui::Button::new("Upscale Video")).clicked() {
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
            egui::ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
                for line in &self.log {
                    ui.monospace(line);
                }
            });
        });
        if busy {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
    }
}
