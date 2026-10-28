# vapourkit-rs

A native Rust port of [Vapourkit](https://github.com/Kim2091/vapourkit), the VapourSynth video upscaling app, without Electron. GUI is [egui](https://github.com/emilk/egui).

Status: early. You pick a video and an output, build a filter chain from AI models (TensorRT, DirectML or NCNN) and VapourSynth filters (161 bundled `.vkfilter` templates, your own `.vkfilter` files, or code typed in), optionally edit the generated script by hand, set up the encode (H.264, H.265, AV1 or ProRes on CPU, NVENC, AMF or QSV, or your own ffmpeg arguments, saved as named presets), and it runs `vspipe` into `ffmpeg` with progress and a live preview of the output. VapourSynth, vs-mlrt, a source plugin (BestSource, or L-SMASH / FFMS2) and ffmpeg/ffprobe must already be installed and on `PATH`; first-run setup comes later. Files can be dropped onto the window or passed on the command line (`vapourkit video.mkv flow.vkworkflow`).

```sh
cargo run --release -p vk-gui            # the app
cargo run -p vk-core --example upscale -- in.mkv model.onnx out.mkv   # headless
```

## Layout

- `crates/vk-core`: filter chain and .vkfilter (`chain.rs`), script generation (`script.rs`), encoding (`encoding.rs`), .vkworkflow (`workflow.rs`) and the vspipe → ffmpeg job (`job.rs`)
- `crates/vk-gui`: the egui app
- `assets/`: `vapoursynth_template.vpy`, `vk_runtime.py` and `filters/*.vkfilter`, taken unchanged from upstream

## Roadmap

1. ~~Skeleton: video + model → vspipe | ffmpeg with progress~~
2. ~~Encoding settings and saved presets~~ (MIGraphX still to do)
3. First-run setup (Python venv, VapourSynth, plugins, models)
4. ~~Filter chains, `.vkfilter` / `.vkworkflow`, editable script~~ (stage references and Load Video side chains still to do)
5. ~~Live preview~~, side-by-side compare
6. Queue/batch, TensorRT engine builds, update check
7. Colour grading, LUTs, crop, DLSS, Discord presence, installers

## License

GPL-3.0, as upstream.
