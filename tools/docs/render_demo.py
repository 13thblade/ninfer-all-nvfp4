from pathlib import Path
import json
import shutil
import subprocess
import textwrap

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "benchmark-results"
WORK = BUILD / "readme-demo"
FFMPEG = Path(r"C:/Program Files/ShareX/ffmpeg.exe")
FPS = 30
DURATION = 13
REPLAY_START = 2.5
ILLUSTRATIVE_RATE = 125

WORK.mkdir(parents=True, exist_ok=True)
data = json.loads((ROOT / "docs/assets/ninfer-demo-recording.json").read_text())
sample = data["sample"]
events = sample["events"]
assets = ROOT / "docs/assets"

def stamp(seconds):
    count = round(seconds * 100)
    return f"{count // 360000}:{count // 6000 % 60:02}:{count // 100 % 60:02}.{count % 100:02}"

def escape(text):
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")")

lines = ["""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Main,Segoe UI,32,&H00F6F6F0,&H00000000,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""]

def label(text, x, y, size=32, color="F0F6F0", start=0, end=DURATION, bold=False):
    tags = f"{{\\pos({x},{y})\\fs{size}\\c&H{color}&\\b{int(bold)}}}"
    lines.append(f"Dialogue: 0,{stamp(start)},{stamp(end)},Main,,0,0,0,,{tags}{text}\n")

label("NINFER-3090 / INFERENCE", 120, 64, 23, "99FF83", bold=True)
label("Qwen3.8-27B", 115, 110, 58, bold=True)
label("RTX 3090  /  DFlash2 K=7  /  rk4v4 KV  /  Native C++ / CUDA", 120, 191, 27, "A2B19C")
label("READY", 120, 258, 20, "99FF83", end=REPLAY_START)
illustrated_seconds = 155 / ILLUSTRATIVE_RATE
label("GENERATING", 120, 258, 20, "99FF83", start=REPLAY_START, end=REPLAY_START + illustrated_seconds)
label("COMPLETE", 120, 258, 20, "99FF83", start=REPLAY_START + illustrated_seconds)
label(f"{ILLUSTRATIVE_RATE} tok/s", 120, 892, 54, "99FF83", bold=True)
label("illustrative rate", 120, 960, 21, "A2B19C")
label("DFlash2 K=7", 760, 910, 36, "99FF83", bold=True)
label("RTX 3090 / 24 GB", 1330, 910, 30, "99FF83", bold=True)
label("Illustrative animation / not a benchmark recording", 120, 1020, 21, "A2B19C")

text = ""
for index, event in enumerate(events):
    text += event["text"]
    paragraphs = []
    for paragraph in text.splitlines():
        paragraphs.extend(textwrap.wrap(paragraph, width=96) or [""])
    shown = "\\N".join(escape(line) for line in paragraphs)
    timing_scale = illustrated_seconds / sample["wall_seconds"]
    start = REPLAY_START + event["seconds"] * timing_scale
    end = REPLAY_START + events[index + 1]["seconds"] * timing_scale if index + 1 < len(events) else DURATION
    if end > start:
        label(shown, 120, 310, 30, start=start, end=end)

(WORK / "demo.ass").write_text("".join(lines), encoding="utf-8-sig")
filters = ("drawbox=x=70:y=35:w=1780:h=970:color=0x101612:t=fill,"
           "drawbox=x=110:y=240:w=1700:h=2:color=0x25432e:t=fill,"
           "drawbox=x=110:y=850:w=1700:h=2:color=0x25432e:t=fill,subtitles=demo.ass")
subprocess.run([str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                f"color=c=0x070a08:s=1920x1080:r={FPS}:d={DURATION}", "-vf", filters,
                "-c:v", "libx264", "-preset", "fast", "-crf", "21", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "demo.mp4"], cwd=WORK, check=True, timeout=180)
subprocess.run([str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y", "-i", "demo.mp4",
                "-filter_complex", "fps=12,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
                "-loop", "0", "demo.gif"], cwd=WORK, check=True, timeout=180)
for suffix in ("mp4", "gif"):
    shutil.copyfile(WORK / f"demo.{suffix}", assets / f"ninfer-demo.{suffix}")

poster = '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="640" viewBox="0 0 1280 640" role="img" aria-label="NInfer-3090 Qwen3.8-27B recorded inference, DFlash2 K=7, RTX 3090">\n<rect width="1280" height="640" rx="20" fill="#070a08"/><rect x="40" y="35" width="1200" height="565" rx="14" fill="#101612" stroke="#25432e"/>\n<g font-family="Segoe UI,Arial,sans-serif"><text x="80" y="95" font-size="20" fill="#83ff99">NINFER-3090 / RECORDED INFERENCE</text><text x="77" y="185" font-size="60" font-weight="700" fill="#f0f6f0">Qwen3.8-27B</text><text x="80" y="242" font-size="25" fill="#9cb1a2">RTX 3090 / DFlash2 K=7 / rk4v4 KV</text><path d="M80 280h1120" stroke="#25432e"/><circle cx="130" cy="375" r="43" fill="#83ff99"/><path d="M120 352l30 23-30 23z" fill="#070a08"/><text x="205" y="385" font-size="34" fill="#f0f6f0">Play recorded inference at 1x</text><text x="80" y="500" font-size="31" fill="#83ff99">155 tokens / 1.71 s / 99.4 tok/s decode</text><text x="80" y="555" font-size="20" fill="#9cb1a2">One illustrative request. Model already loaded. No speed-up.</text></g></svg>'
poster = poster.replace('recorded inference', 'inference illustration').replace('RECORDED INFERENCE', 'INFERENCE').replace('Play recorded inference at 1x', 'Native C++ / CUDA inference').replace('155 tokens / 1.71 s / 99.4 tok/s decode', '125 tok/s / DFlash2 K=7').replace('One illustrative request. Model already loaded. No speed-up.', 'Illustrative animation / not a benchmark recording')
(assets / "ninfer-demo-poster.svg").write_text(poster, encoding="utf-8")
subprocess.run([str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y", "-ss", "5", "-i", "demo.mp4", "-frames:v", "1", "preview.png"], cwd=WORK, check=True)
print("Created illustrative animation with CPU rendering; original capture data preserved")
