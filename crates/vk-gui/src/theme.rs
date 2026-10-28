//! Colours and widgets that give the app the look of upstream Vapourkit:
//! near-black panels, teal accent, uppercase section headers.

use eframe::egui::{self, Color32, CornerRadius, FontId, Margin, RichText, Stroke, TextStyle};

pub const ACCENT: Color32 = Color32::from_rgb(45, 212, 191);
pub const ACCENT_DIM: Color32 = Color32::from_rgb(20, 70, 64);
pub const DANGER: Color32 = Color32::from_rgb(239, 83, 80);
pub const BG: Color32 = Color32::from_rgb(17, 19, 22);
pub const PANEL: Color32 = Color32::from_rgb(22, 25, 29);
pub const CARD: Color32 = Color32::from_rgb(28, 32, 37);
pub const BORDER: Color32 = Color32::from_rgb(44, 50, 57);
pub const TEXT: Color32 = Color32::from_rgb(220, 224, 229);
pub const WEAK: Color32 = Color32::from_rgb(140, 148, 158);

pub fn apply(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.text_styles = [
        (TextStyle::Small, FontId::proportional(12.0)),
        (TextStyle::Body, FontId::proportional(14.0)),
        (TextStyle::Button, FontId::proportional(14.0)),
        (TextStyle::Heading, FontId::proportional(18.0)),
        (TextStyle::Monospace, FontId::monospace(13.0)),
    ]
    .into();
    style.spacing.item_spacing = egui::vec2(8.0, 8.0);
    style.spacing.button_padding = egui::vec2(10.0, 5.0);
    style.spacing.interact_size.y = 26.0;

    let v = &mut style.visuals;
    *v = egui::Visuals::dark();
    v.panel_fill = PANEL;
    v.window_fill = PANEL;
    v.extreme_bg_color = BG;
    v.faint_bg_color = CARD;
    v.override_text_color = Some(TEXT);
    v.window_stroke = Stroke::new(1.0_f32, BORDER);
    v.selection.bg_fill = ACCENT_DIM;
    v.selection.stroke = Stroke::new(1.0_f32, ACCENT);
    v.hyperlink_color = ACCENT;
    v.window_corner_radius = CornerRadius::same(8);
    v.menu_corner_radius = CornerRadius::same(6);
    let radius = CornerRadius::same(5);
    for (w, fill) in [
        (&mut v.widgets.noninteractive, CARD),
        (&mut v.widgets.inactive, Color32::from_rgb(36, 41, 47)),
        (&mut v.widgets.hovered, Color32::from_rgb(46, 52, 60)),
        (&mut v.widgets.active, Color32::from_rgb(54, 61, 70)),
        (&mut v.widgets.open, Color32::from_rgb(46, 52, 60)),
    ] {
        w.bg_fill = fill;
        w.weak_bg_fill = fill;
        w.corner_radius = radius;
    }
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, BORDER);
    v.widgets.inactive.bg_stroke = Stroke::new(1.0_f32, BORDER);
    v.widgets.hovered.bg_stroke = Stroke::new(1.0_f32, ACCENT);
    v.widgets.active.bg_stroke = Stroke::new(1.0_f32, ACCENT);
    ctx.set_style(style);
}

/// A section title with the accent stripe on its left, as upstream's
/// PREVIEW / SOURCE / FILTERS headers, and controls on the right.
pub fn section(ui: &mut egui::Ui, title: &str, note: &str, right: impl FnOnce(&mut egui::Ui)) {
    ui.horizontal(|ui| {
        let (rect, _) = ui.allocate_exact_size(egui::vec2(3.0, 20.0), egui::Sense::hover());
        ui.painter().rect_filled(rect, 1.0, ACCENT);
        ui.label(RichText::new(title).strong().size(14.0).extra_letter_spacing(1.6));
        if !note.is_empty() {
            ui.label(RichText::new(note).color(WEAK).small());
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), right);
    });
}

pub fn card() -> egui::Frame {
    egui::Frame::new()
        .fill(CARD)
        .stroke(Stroke::new(1.0_f32, BORDER))
        .corner_radius(6)
        .inner_margin(Margin::symmetric(10, 8))
}

/// The filled teal button for the main action.
pub fn primary(text: &str) -> egui::Button<'static> {
    egui::Button::new(RichText::new(text.to_string()).color(BG).strong()).fill(ACCENT)
}

pub fn danger(text: &str) -> egui::Button<'static> {
    egui::Button::new(RichText::new(text.to_string()).color(DANGER)).stroke(Stroke::new(1.0_f32, DANGER))
}

/// Small uppercase field label, as upstream's FORMAT / SAVE TO.
pub fn field_label(ui: &mut egui::Ui, text: &str) -> egui::Response {
    ui.label(RichText::new(text.to_uppercase()).small().color(WEAK).extra_letter_spacing(1.0))
}
