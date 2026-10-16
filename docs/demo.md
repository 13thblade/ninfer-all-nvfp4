# README animation

The README uses an original black-and-green inference illustration. Its **125 tokens/s display
is illustrative**, not a new measurement. Text from the earlier capture is animated at a designed
pace; the animation labels this directly. See the README performance section for sourced results.

The original prompt, response and unchanged measurements remain in
[the recording data](assets/ninfer-demo-recording.json). They are separate from this illustration.

Run `uv run tools/docs/render_demo.py` to recreate the assets with Python's standard library and
the installed FFmpeg specified at the top of the script. Rendering uses the CPU; it does not load
a model or use the GPU. Scratch outputs go into `benchmark-results/readme-demo/`.
The MP4 is 1920 by 1080 at 30 fps; the GIF is 960 by 540 at 12 fps. Both run for 13 seconds.
