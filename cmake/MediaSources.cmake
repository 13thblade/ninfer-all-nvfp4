# Source releases of the media decoders that Linux builds link statically (cmake/NinferFFmpeg.cmake).
# scripts/fetch-media-sources.sh downloads them and checks these digests; the build itself never
# downloads anything. FFmpeg's archive is verified against the FFmpeg release signing key
# FCF986EA15E6E293A5644F10B4322F04D67658D8 and dav1d's against the VideoLAN release signing key
# 65F7C6B4206BD057A7EB73787180713BE58D1ADC before a digest is pinned here.
set(NINFER_FFMPEG_VERSION 9.0.2)
set(NINFER_FFMPEG_ARCHIVE ffmpeg-${NINFER_FFMPEG_VERSION}.tar.xz)
set(NINFER_FFMPEG_URL "https://ffmpeg.org/releases/${NINFER_FFMPEG_ARCHIVE}")
set(NINFER_FFMPEG_SHA256 8c3850283eb25fa026482078a04051e0be17347b09ef81a0849bec15a96e002e)
set(NINFER_FFMPEG_LICENSE ffmpeg-${NINFER_FFMPEG_VERSION}/COPYING.LGPLv2.1)

set(NINFER_DAV1D_VERSION 1.5.4)
set(NINFER_DAV1D_ARCHIVE dav1d-${NINFER_DAV1D_VERSION}.tar.xz)
set(NINFER_DAV1D_URL
    "https://downloads.videolan.org/pub/videolan/dav1d/${NINFER_DAV1D_VERSION}/${NINFER_DAV1D_ARCHIVE}")
set(NINFER_DAV1D_SHA256 686616b7c69eb88d44459391ab25cac13b6647a3b288835c5784e71c1514a5c5)
set(NINFER_DAV1D_LICENSE dav1d-${NINFER_DAV1D_VERSION}/COPYING)

set(NINFER_MEDIA_SOURCES FFMPEG DAV1D)
