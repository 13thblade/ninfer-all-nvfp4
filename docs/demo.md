# A real response, in an original presentation

The [README demo](assets/ninfer-demo.mp4) is a designed replay of a response captured from this
repository's native Windows server. The animation uses the actual arrival time of every streamed
content chunk, at 1x speed. It is not a screen recording of an included chat application.

The framing holds for 2.5 seconds before replay begins and holds the completed output afterward.
That reading time does not change the generated-text timing. The model was loaded before the
request; its cold startup is not shown. The frame shows model and execution settings rather than a chat application. The exact prompt,
full response, chunk timestamps and metrics are in [the recording data](assets/ninfer-demo-recording.json).

## This recording

- 16 October 2026; one RTX 3090, native Windows, CUDA 13.2, MSVC 14.43.
- Qwen3.8-27B, pinned public artifact revision `d47f2732d369acaec76dc44228f67c72b081df2f`.
- DFlash2 K=7, one lane, `rk4v4` KV, 16K context cap, thinking and prefix reuse off.
- 54 input tokens; 155 generated tokens, ending normally.
- 1.708 seconds for the request; 158.6 ms to first streamed content; server-reported decode 99.4 tokens/s.
- One illustrative request, without warmups. It is separate from the six-sample README benchmark.

The response is model-generated prose, not a security certification. The server can intentionally
use the network for remote image requests or optional storage when those features are used.

## How it is made

The renderer uses Python's standard library and an installed FFmpeg with libass. It places the
recorded text into an original black-and-green inference layout, exports an H.264 MP4, and creates a smaller
animated GIF for GitHub's README. It does not borrow HyperQwen's video, artwork, timings or numbers.

Run `uv run tools/docs/render_demo.py` from the checkout after setting the FFmpeg constant at the
top of that script. Outputs live under `benchmark-results/readme-demo/` and `docs/assets/`.

The MP4 is 1920 × 1080 at 30 fps for 13 seconds. The GIF is 960 × 540 at 12 fps.
The original SVG hero and poster remain editable vector assets.
