# Third-party notices

The project MIT license covers this project's code. It does not relicense its
dependencies, fonts, user-supplied media or other third-party material.

## Bundled fonts

VT323, Share Tech Mono and Press Start 2P are distributed under the SIL Open Font
License 1.1. Each font's original copyright and license notice is included beside
the font in `src/vhs_broadcast/assets/fonts/`:

- `VT323-OFL.txt`
- `ShareTechMono-OFL.txt`
- `PressStart2P-OFL.txt`

Source: [Google Fonts](https://github.com/google/fonts), directories `ofl/vt323`,
`ofl/sharetechmono`, and `ofl/pressstart2p`.

VCR OSD Mono is an optional user-installed font, excluded from source tracking and
distribution builds. Its usage and redistribution terms are separate; the MIT
license grants no rights to it. The optional fetch script is not run on startup.

## Dependencies

Python dependencies are declared in `pyproject.toml` and resolved in `uv.lock`.
They retain their upstream licenses, including the license obligations of Qt/
PySide6 and the codec libraries used by OpenCV and PyAV. Dependency distributions
carry their own notices. Review those licenses when bundling a standalone binary;
installing this source project is not a blanket MIT grant over its dependencies.

The console and analog looks are original aesthetic implementations. Product
names describe visual inspiration and do not imply affiliation or endorsement.
