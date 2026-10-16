from pathlib import Path
import html
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

def label(text, x, y, size=32, color="F6F6F0", start=0, end=DURATION, bold=False):
    tags = f"{{\\pos({x},{y})\\fs{size}\\c&H{color}&\\b{int(bold)}}}"
    lines.append(f"Dialogue: 0,{stamp(start)},{stamp(end)},Main,,0,0,0,,{tags}{text}\n")

label("NINFER / RTX 3090", 100, 83, 24, "AAFA75", bold=True)
label("An assistant.\\NOn your desk.", 92, 190, 76, bold=True)
label("A real Qwen3.8-27B response.\\NOne GPU. Native Windows.", 100, 406, 29, "D1BEAA")
label("YOUR CONVERSATION, LOCAL", 100, 588, 19, "AAFA75", bold=True)
label("Privacy. Offline use after setup.\\NControl over how you run it.", 100, 628, 29)
label("No cloud inference account required.", 100, 735, 24, "D1BEAA")
label("LOCAL MODEL  /  QWEN3.8-27B", 798, 110, 22, "AAFA75", bold=True)
label("Why run an AI assistant locally?", 798, 172, 36, bold=True)
label("A conversation on your own machine.", 800, 235, 20, "D1BEAA")
label("MODEL READY", 800, 325, 19, "AAFA75", end=REPLAY_START)
label("GENERATING", 800, 325, 19, "AAFA75", start=REPLAY_START, end=REPLAY_START + sample["wall_seconds"])
label("COMPLETE", 800, 325, 19, "AAFA75", start=REPLAY_START + sample["wall_seconds"])
label("155 output tokens", 800, 862, 29, bold=True)
label("1.71 s request", 1220, 862, 29, bold=True)
label("99.4 tokens/s decode", 800, 911, 23, "D1BEAA")
label("159 ms first streamed text", 1220, 911, 23, "D1BEAA")
label("Recorded output replayed at 1x  |  Model already loaded  |  DFlash2 K=7  |  16 October 2026", 100, 1000, 22, "D1BEAA")

text = ""
for index, event in enumerate(events):
    text += event["text"]
    paragraphs = []
    for paragraph in text.splitlines():
        paragraphs.extend(textwrap.wrap(paragraph, width=69) or [""])
    shown = "\\N".join(escape(line) for line in paragraphs)
    start = REPLAY_START + event["seconds"]
    end = REPLAY_START + events[index + 1]["seconds"] if index + 1 < len(events) else DURATION
    if end > start:
        label(shown, 800, 380, 27, start=start, end=end)

(WORK / "demo.ass").write_text("".join(lines), encoding="utf-8-sig")
filters = ("drawbox=x=740:y=70:w=1090:h=900:color=0x112738:t=fill,"
           "drawbox=x=780:y=300:w=1000:h=2:color=0x294356:t=fill,"
           "drawbox=x=780:y=825:w=1000:h=2:color=0x294356:t=fill,subtitles=demo.ass")
subprocess.run([str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                f"color=c=0x07162b:s=1920x1080:r={FPS}:d={DURATION}", "-vf", filters,
                "-c:v", "libx264", "-preset", "fast", "-crf", "21", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "demo.mp4"], cwd=WORK, check=True, timeout=180)
subprocess.run([str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y", "-i", "demo.mp4",
                "-filter_complex", "fps=12,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
                "-loop", "0", "demo.gif"], cwd=WORK, check=True, timeout=180)
for suffix in ("mp4", "gif"):
    shutil.copyfile(WORK / f"demo.{suffix}", assets / f"ninfer-demo.{suffix}")

poster = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="640" viewBox="0 0 1280 640" role="img" aria-label="Play a recorded NInfer response on RTX 3090">
<rect width="1280" height="640" rx="24" fill="#07162b"/><rect x="650" y="65" width="560" height="510" rx="20" fill="#112738"/>
<g font-family="Segoe UI,Arial,sans-serif"><text x="65" y="100" font-size="20" fill="#75faaa">NINFER / RTX 3090</text><text x="60" y="210" font-size="65" font-weight="700" fill="#f0f6f6">Watch it answer.</text><text x="65" y="280" font-size="27" fill="#aabed1">Qwen3.8-27B. One card.</text><circle cx="130" cy="410" r="50" fill="#75faaa"/><path d="M118 385l34 25-34 25z" fill="#07162b"/><text x="205" y="420" font-size="26" fill="#f0f6f6">Play the real-time replay</text><text x="70" y="545" font-size="20" fill="#aabed1">Real output · Recorded arrival times · 1x speed</text><text x="690" y="125" font-size="22" fill="#75faaa">LOCAL ASSISTANT</text><text x="690" y="200" font-size="28" fill="#f0f6f6">{html.escape(sample['text'].splitlines()[0][:36])}</text><text x="690" y="285" font-size="26" fill="#f0f6f6">Privacy. Offline use. Control.</text><text x="690" y="420" font-size="42" fill="#75faaa">155 tokens / 1.71 s</text><text x="690" y="475" font-size="22" fill="#aabed1">Model already loaded · Windows</text></g></svg>'''
(assets / "ninfer-demo-poster.svg").write_text(poster, encoding="utf-8")
public_capture = {"prompt": data["prompt"], "sample": sample, "source_commit": "931bd49ba332f48fad4b5230eec5a95ec5701290", "presentation": {"response_replay_start_seconds": REPLAY_START, "speed": 1, "duration_seconds": DURATION}}
(ROOT / "docs/assets/ninfer-demo-recording.json").write_text(json.dumps(public_capture, indent=2), encoding="utf-8")
subprocess.run([str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y", "-ss", "5", "-i", "demo.mp4", "-frames:v", "1", "preview.png"], cwd=WORK, check=True)
print("Created original MP4, animated GIF, SVG poster and exact response-arrival data")
