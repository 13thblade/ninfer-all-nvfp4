#!/usr/bin/env bash
set -euo pipefail

# Writes the media fixtures that tests/test_media_decode.cpp decodes. Needs python3 and an FFmpeg
# with the encoders named below (libx264, libx265, libaom-av1, libvpx, libwebp, libopus); FFMPEG
# names another ffmpeg executable. The test checks the decoded pixels against the colors written
# here, not against this encoder's output.
#
#   clip:      6 frames of 64x48 at 6 fps, frame i one solid color (COLORS in the test)
#   pixels:    one 4x2 image of eight distinct colors (PIXELS in the test)
#   quadrants: one 64x48 image of four solid quadrants (QUADRANTS in the test)

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
ffmpeg=("${FFMPEG:-ffmpeg}" -hide_banner -loglevel error -y)
bitexact=(-map_metadata -1 -fflags +bitexact -flags:v +bitexact -flags:a +bitexact)

raw() {
    python3 - "$1" <<'PY'
import sys
out = sys.stdout.buffer
kind = sys.argv[1]
if kind == 'clip':
    for color in [(200, 40, 40), (40, 200, 40), (40, 40, 200),
                  (200, 200, 40), (40, 200, 200), (200, 40, 200)]:
        out.write(bytes(color) * (64 * 48))
elif kind == 'pixels':
    out.write(bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128,
                     1, 2, 3, 250, 251, 252, 7, 8, 9, 200, 150, 100]))
else:
    quadrants = [(200, 40, 40), (40, 200, 40), (40, 40, 200), (200, 200, 40)]
    for y in range(48):
        for x in range(64):
            out.write(bytes(quadrants[(y >= 24) * 2 + (x >= 32)]))
PY
}

clip=(-f rawvideo -pix_fmt rgb24 -s 64x48 -r 6 -i -)
tone=(-f lavfi -i 'sine=frequency=440:sample_rate=8000:duration=1' -shortest -ac 1 -b:a 16k)
pixels=(-f rawvideo -pix_fmt rgb24 -s 4x2 -i -)
quadrants=(-f rawvideo -pix_fmt rgb24 -s 64x48 -i -)
palette='split[a][b];[a]palettegen=max_colors=8:reserve_transparent=0:stats_mode=full[p];[b][p]paletteuse=dither=none'

# Decoded formats; two of the clips carry an audio track, which decoding skips.
raw clip | "${ffmpeg[@]}" "${clip[@]}" "${tone[@]}" -c:v libx264 -crf 10 -pix_fmt yuv420p \
    -c:a aac "${bitexact[@]}" h264-aac.mp4
raw clip | "${ffmpeg[@]}" "${clip[@]}" -c:v libx265 -crf 10 -pix_fmt yuv420p -tag:v hvc1 \
    -x265-params log-level=error "${bitexact[@]}" hevc.mov
raw clip | "${ffmpeg[@]}" "${clip[@]}" -c:v mpeg4 -q:v 2 -pix_fmt yuv420p "${bitexact[@]}" mpeg4.mp4
raw clip | "${ffmpeg[@]}" "${clip[@]}" "${tone[@]}" -c:v libaom-av1 -crf 20 -b:v 0 -cpu-used 8 \
    -pix_fmt yuv420p -c:a libopus "${bitexact[@]}" av1-opus.webm
raw clip | "${ffmpeg[@]}" "${clip[@]}" -c:v libvpx -crf 4 -b:v 1M -pix_fmt yuv420p "${bitexact[@]}" vp8.webm
raw clip | "${ffmpeg[@]}" "${clip[@]}" -c:v libvpx-vp9 -crf 10 -b:v 0 -pix_fmt yuv420p \
    "${bitexact[@]}" vp9.mkv
raw clip | "${ffmpeg[@]}" "${clip[@]}" -vf "$palette" "${bitexact[@]}" animation.gif
raw pixels | "${ffmpeg[@]}" "${pixels[@]}" -c:v png -pix_fmt rgb24 "${bitexact[@]}" pixels.png
raw pixels | "${ffmpeg[@]}" "${pixels[@]}" -c:v bmp -pix_fmt bgr24 "${bitexact[@]}" pixels.bmp
raw pixels | "${ffmpeg[@]}" "${pixels[@]}" -c:v libwebp -lossless 1 "${bitexact[@]}" pixels.webp
raw pixels | "${ffmpeg[@]}" "${pixels[@]}" -vf "$palette" "${bitexact[@]}" pixels.gif
raw quadrants | "${ffmpeg[@]}" "${quadrants[@]}" -c:v libwebp -quality 90 "${bitexact[@]}" quadrants.webp

# Formats outside the list: a container, a codec in a listed container, and an image.
raw clip | "${ffmpeg[@]}" "${clip[@]}" -c:v mjpeg -q:v 4 -pix_fmt yuvj420p "${bitexact[@]}" mjpeg.avi
raw clip | "${ffmpeg[@]}" "${clip[@]}" -c:v ffv1 -pix_fmt yuv420p "${bitexact[@]}" ffv1.mkv
raw pixels | "${ffmpeg[@]}" "${pixels[@]}" -c:v tiff "${bitexact[@]}" pixels.tiff
