#!/usr/bin/env python3
"""Fetch the VCR OSD Mono font into the app's font folder.

VCR OSD Mono (by Riciery Leal) is the classic on-screen-display face from old
VCRs, and looks great for the OSD/clock overlay. It is a THIRD-PARTY font with
its own license (listed as free on dafont -- see the page below), so it is NOT
bundled with this project. The run script offers to fetch it on first launch;
you can also run this directly. It drops the .ttf into
src/vhs_broadcast/assets/fonts/ and the app picks it up in the FONT dropdown
automatically (it also becomes the default OSD font).

Usage:
    python scripts/fetch_vcr_font.py            # prompts before downloading
    python scripts/fetch_vcr_font.py --yes      # no prompt (e.g. CI/dotfiles)
    python scripts/fetch_vcr_font.py --zip FILE # install from a local zip
    python scripts/fetch_vcr_font.py --if-missing  # launcher mode (see below)
"""
from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

PAGE = "https://www.dafont.com/vcr-osd-mono.font"
URL = "https://dl.dafont.com/dl/?f=vcr_osd_mono"
DEST = (Path(__file__).resolve().parent.parent
        / "src" / "vhs_broadcast" / "assets" / "fonts")
_UA = "Mozilla/5.0 (compatible; vhs-broadcast-font-fetch/1.0)"


def download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def extract_ttfs(zip_bytes: bytes, dest: Path) -> int:
    n = 0
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for name in z.namelist():
            if name.lower().endswith(".ttf"):
                out = dest / Path(name).name
                out.write_bytes(z.read(name))
                print("  +", out.name)
                n += 1
    return n


def already_present(dest: Path):
    for p in dest.glob("*.ttf"):
        if "vcr" in p.stem.lower():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch VCR OSD Mono into the OSD "
                                             "font folder.")
    ap.add_argument("--yes", "-y", action="store_true", help="skip the prompt")
    ap.add_argument("--zip", help="install from a local .zip instead of "
                                  "downloading")
    ap.add_argument("--if-missing", action="store_true",
                    help="launcher mode: silent if the font is present or was "
                         "previously declined; otherwise prompt once")
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    skip_marker = DEST / ".vcr-skip"
    have = already_present(DEST)
    if have:
        if not args.if_missing:
            print(f"VCR OSD Mono already installed: {have.name}")
        return 0
    if args.if_missing and skip_marker.exists():
        return 0  # user declined before; don't nag on every launch

    if args.zip:
        data = Path(args.zip).read_bytes()
    else:
        print(__doc__)
        print("Source page:", PAGE)
        if not args.yes:
            try:
                ans = input("Download VCR OSD Mono now? [y/N] ").strip().lower()
            except EOFError:
                ans = ""
            if ans not in ("y", "yes"):
                print("Skipped. You can also download it manually from the page "
                      f"above and drop the .ttf into:\n  {DEST}")
                if args.if_missing:
                    skip_marker.write_text(
                        "You declined the VCR OSD Mono download from the "
                        "launcher.\nRun  python scripts/fetch_vcr_font.py  (or "
                        "delete this file) to be offered it again.\n")
                    print("(Won't ask again on launch.)")
                return 0 if args.if_missing else 1
        print("Downloading...")
        try:
            data = download(URL)
        except Exception as e:  # noqa: BLE001
            print("Download failed:", e)
            print("Download it manually from", PAGE,
                  "and drop the .ttf into:\n ", DEST)
            print("...or re-run with:  --zip path/to/vcr_osd_mono.zip")
            return 2

    try:
        n = extract_ttfs(data, DEST)
    except zipfile.BadZipFile:
        print("That didn't look like a zip archive. Grab it from", PAGE)
        return 3
    if n == 0:
        print("No .ttf found in the archive.")
        return 3
    print(f"Done - added {n} font file(s). Restart the app and pick it in the "
          "FONT dropdown (it's now the default OSD font).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
