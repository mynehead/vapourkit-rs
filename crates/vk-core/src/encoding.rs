//! Video encoding settings and the ffmpeg arguments they resolve to, as in
//! electron/ffmpegConfig.ts and electron/encodingSettings.ts.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Codec {
    H264,
    H265,
    Av1,
    ProRes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Encoder {
    Software,
    Nvidia,
    Amd,
    Intel,
}

impl Codec {
    pub const ALL: [Codec; 4] = [Codec::H264, Codec::H265, Codec::Av1, Codec::ProRes];

    pub fn label(self) -> &'static str {
        match self {
            Codec::H264 => "H.264",
            Codec::H265 => "H.265 / HEVC",
            Codec::Av1 => "AV1",
            Codec::ProRes => "ProRes",
        }
    }

    pub fn encoders(self) -> &'static [Encoder] {
        match self {
            Codec::ProRes => &[Encoder::Software],
            _ => &[Encoder::Software, Encoder::Nvidia, Encoder::Amd, Encoder::Intel],
        }
    }

    pub fn has_quality(self) -> bool {
        self != Codec::ProRes
    }
}

impl Encoder {
    pub fn label(self) -> &'static str {
        match self {
            Encoder::Software => "Software (CPU)",
            Encoder::Nvidia => "NVIDIA NVENC",
            Encoder::Amd => "AMD AMF",
            Encoder::Intel => "Intel QSV",
        }
    }

    pub fn quality_label(self) -> &'static str {
        match self {
            Encoder::Software => "CRF",
            Encoder::Nvidia => "CQ",
            Encoder::Amd => "QP",
            Encoder::Intel => "ICQ",
        }
    }
}

fn ffmpeg_encoder(codec: Codec, encoder: Encoder) -> &'static str {
    use {Codec::*, Encoder::*};
    match (codec, encoder) {
        (H264, Software) => "libx264",
        (H264, Nvidia) => "h264_nvenc",
        (H264, Amd) => "h264_amf",
        (H264, Intel) => "h264_qsv",
        (H265, Software) => "libx265",
        (H265, Nvidia) => "hevc_nvenc",
        (H265, Amd) => "hevc_amf",
        (H265, Intel) => "hevc_qsv",
        (Av1, Software) => "libsvtav1",
        (Av1, Nvidia) => "av1_nvenc",
        (Av1, Amd) => "av1_amf",
        (Av1, Intel) => "av1_qsv",
        (ProRes, _) => "prores_ks",
    }
}

pub fn presets(codec: Codec, encoder: Encoder) -> &'static [&'static str] {
    const X26X: &[&str] = &["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"];
    match (codec, encoder) {
        (Codec::ProRes, _) => &[],
        (_, Encoder::Nvidia) => &["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
        (_, Encoder::Amd) => &["speed", "balanced", "quality"],
        (_, Encoder::Intel) => &X26X[2..],
        (Codec::Av1, Encoder::Software) => &["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
        _ => X26X,
    }
}

pub fn default_preset(codec: Codec, encoder: Encoder) -> &'static str {
    match (codec, encoder) {
        (Codec::Av1, Encoder::Software) => "8",
        (_, Encoder::Software) => "medium",
        (_, Encoder::Nvidia) => "p7",
        (_, Encoder::Amd) => "quality",
        (_, Encoder::Intel) => "veryslow",
    }
}

/// (min, max, default) on the encoder's quality scale; lower is better.
pub fn quality_range(codec: Codec, encoder: Encoder) -> (u32, u32, u32) {
    use {Codec::*, Encoder::*};
    match (codec, encoder) {
        (H264, Software) => (0, 51, 18),
        (H265, Software) => (0, 51, 20),
        (Av1, Software) => (0, 63, 25),
        (Av1, Nvidia) => (1, 63, 25),
        (Av1, Amd) => (0, 255, 100),
        (Av1, Intel) => (1, 51, 25),
        (_, Nvidia | Intel) => (1, 51, 20),
        (_, Amd) => (0, 51, 20),
        (ProRes, Software) => (0, 51, 18),
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Encoding {
    /// Use `custom_args` instead of the simple controls; both are kept.
    pub advanced: bool,
    pub codec: Codec,
    pub encoder: Encoder,
    pub preset: String,
    pub quality: u32,
    pub custom_args: String,
}

impl Default for Encoding {
    fn default() -> Self {
        Encoding {
            advanced: false,
            codec: Codec::H264,
            encoder: Encoder::Software,
            preset: "medium".into(),
            quality: 18,
            custom_args: String::new(),
        }
    }
}

impl Encoding {
    /// Keeps encoder, preset and quality valid after codec or encoder changed.
    pub fn normalize(&mut self) {
        if !self.codec.encoders().contains(&self.encoder) {
            self.encoder = Encoder::Software;
        }
        let presets = presets(self.codec, self.encoder);
        if !presets.is_empty() && !presets.contains(&self.preset.as_str()) {
            self.preset = default_preset(self.codec, self.encoder).into();
        }
        let (min, max, _) = quality_range(self.codec, self.encoder);
        self.quality = self.quality.clamp(min, max);
    }

    /// The ffmpeg video arguments for this encoding.
    pub fn args(&self) -> Vec<String> {
        if self.advanced {
            // shortcut: split on whitespace, so quoted values with spaces are unsupported; add shell-style splitting if someone needs them.
            return self.custom_args.split_whitespace().map(String::from).collect();
        }
        let q = self.quality.to_string();
        let mut a = vec!["-c:v", ffmpeg_encoder(self.codec, self.encoder)];
        if self.codec.has_quality() {
            a.extend(["-preset", &self.preset]);
            match self.encoder {
                Encoder::Software => a.extend(["-crf", &q]),
                Encoder::Nvidia => a.extend(["-cq", &q, "-b:v", "0"]),
                Encoder::Amd => {
                    a.extend(["-rc", "cqp", "-qp_i", &q, "-qp_p", &q]);
                    if self.codec != Codec::H265 {
                        a.extend(["-qp_b", &q]);
                    }
                }
                Encoder::Intel => a.extend(["-global_quality", &q]),
            }
        }
        a.extend(["-vf", "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709", "-map_metadata", "1"]);
        a.into_iter().map(String::from).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_x264_crf18() {
        assert_eq!(
            Encoding::default().args().join(" "),
            "-c:v libx264 -preset medium -crf 18 -vf setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709 -map_metadata 1"
        );
    }

    #[test]
    fn nvenc_and_amf_quality_args() {
        let mut e = Encoding { codec: Codec::H265, encoder: Encoder::Nvidia, ..Default::default() };
        e.normalize();
        assert!(e.args().join(" ").starts_with("-c:v hevc_nvenc -preset p7 -cq 18 -b:v 0"));
        e.encoder = Encoder::Amd;
        e.normalize();
        let a = e.args().join(" ");
        assert!(a.contains("-qp_p 18") && !a.contains("-qp_b"), "{a}");
    }

    #[test]
    fn prores_has_no_preset_or_quality() {
        let mut e = Encoding { codec: Codec::ProRes, encoder: Encoder::Nvidia, ..Default::default() };
        e.normalize();
        assert_eq!(e.encoder, Encoder::Software);
        assert!(e.args().join(" ").starts_with("-c:v prores_ks -vf"));
    }

    #[test]
    fn advanced_uses_custom_args() {
        let e = Encoding { advanced: true, custom_args: " -c:v libx265  -crf 16 ".into(), ..Default::default() };
        assert_eq!(e.args(), ["-c:v", "libx265", "-crf", "16"]);
    }
}
