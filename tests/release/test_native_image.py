"""Check artifact-to-Docker handoff, including split volumes and incorrect sources."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("native_image", ROOT / ".github/scripts/extract-native-image.py")
IMAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IMAGE)
REVISION = "a" * 40
NAME = "ninfer-linux-x64-cuda13.4-sm75-80-86-89-120a-" + REVISION[:12]


class NativeImageTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "artifact"
        self.source.mkdir()

    def artifact(self, revision=REVISION):
        files = {
            "Dockerfile": b"FROM debian:13-slim\n",
            "bin/multi/ninfer-multicall": b"fixture binary",
            "packages/multi.txt": b"libc6:amd64\n",
            "BUILD-INFO.json": json.dumps(dict(
                source_revision=revision, modified_sources=False, cuda="13.4",
                platform="linux-x64", architectures=["75", "80", "86", "89", "120a"])).encode(),
        }
        files["SHA256SUMS.txt"] = "".join(
            f"{hashlib.sha256(value).hexdigest()}  {name}\n" for name, value in files.items()).encode()
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w:gz") as archive:
            for name, value in files.items():
                info = tarfile.TarInfo(NAME + "/" + name)
                info.size = len(value)
                archive.addfile(info, io.BytesIO(value))
            link = tarfile.TarInfo(NAME + "/bin/multi/ninfer")
            link.type = tarfile.SYMTYPE
            link.linkname = "ninfer-multicall"
            archive.addfile(link)
        data = output.getvalue()
        manifest = []
        for index, start in enumerate(range(0, len(data), 97), 1):
            name = f"{NAME}.tar.gz.{index:03d}"
            chunk = data[start:start + 97]
            (self.source / name).write_bytes(chunk)
            manifest.append(f"{hashlib.sha256(chunk).hexdigest()}  {name}\n")
        (self.source / f"SHA256SUMS-{NAME}.txt").write_text("".join(manifest))

    def test_split_archive_becomes_context_without_recompilation(self):
        self.artifact()
        context = IMAGE.extract(self.source, self.root / "context", REVISION)
        self.assertEqual((context / "bin/multi/ninfer").read_bytes(), b"fixture binary")
        self.assertTrue((context / "Dockerfile").is_file())

    def test_corrupt_volume_is_rejected_before_extraction(self):
        self.artifact()
        (self.source / f"{NAME}.tar.gz.001").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            IMAGE.extract(self.source, self.root / "context", REVISION)
        self.assertFalse((self.root / "context").exists())

    def test_source_mismatch_is_rejected(self):
        self.artifact(revision="b" * 40)
        with self.assertRaisesRegex(ValueError, "metadata"):
            IMAGE.extract(self.source, self.root / "context", REVISION)
