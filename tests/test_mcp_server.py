import asyncio
import base64
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

try:
    from mcp import Client
    from mcp.server.mcpserver.exceptions import ToolError
    from vhs_broadcast import mcp_server as server
except ImportError:  # the mcp extra is optional
    server = None


def write_clip(path, frames=45, size=(320, 240), fps=30):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for i in range(frames):
        frame = np.zeros((size[1], size[0], 3), np.uint8)
        frame[:, :, 1] = 40 + i * 4
        cv2.rectangle(frame, (20 + i * 4, 60), (80 + i * 4, 160), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def decode(image):
    data = image if isinstance(image, bytes) else image.data
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


@unittest.skipIf(server is None, 'mcp extra not installed')
class McpServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.folder.name)
        cls.clip = str(cls.dir / 'clip.mp4')
        write_clip(cls.clip)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_catalog_filters_by_stage_and_search(self):
        full = server.catalog()
        tape = server.catalog(stage='tape')
        self.assertTrue(tape['effects'])
        self.assertTrue(all(e['stage'] == 'tape' for e in tape['effects']))
        self.assertLess(len(tape['effects']), len(full['effects']))
        found = server.catalog(search='DATAMOSH')
        self.assertIn('codec', [e['id'] for e in found['effects']])
        self.assertLess(len(found['effects']), 4)
        brief = server.catalog(stage='camera', summary=True)
        camcorder = next(e for e in brief['effects'] if e['id'] == 'camcorder')
        self.assertIn('camcorder.enabled', camcorder['parameters'])
        self.assertIsInstance(camcorder['parameters'][0], str)
        with self.assertRaisesRegex(ToolError, 'Unknown stage'):
            server.catalog(stage='nope')

    def test_examples_round_trip_through_create_and_validate(self):
        names = [e['name'] for e in server.list_examples()]
        self.assertIn('camcorder-deck', names)
        spec = server.get_example('camcorder-deck')
        output = str(self.dir / 'camcorder.json')
        result = server.create_take(spec, output)
        self.assertEqual(result['path'], str(Path(output).resolve()))
        report = server.validate_take(output)
        self.assertTrue(report['valid'])
        self.assertEqual(report['tracks'], 1)
        with self.assertRaisesRegex(ToolError, 'Output exists'):
            server.create_take(spec, output)

    def test_preview_returns_png_of_requested_width(self):
        spec = dict(server.get_example('block-glitch'), source=self.clip)
        image = asyncio.run(server.preview_frame(spec, time_s=1.0, width=160))
        self.assertEqual(image._mime_type, 'image/png')
        frame = decode(image)
        self.assertEqual(frame.shape[:2], (120, 160))
        strip = asyncio.run(server.preview_strip(spec, [1.2, 0.2], width=160))
        self.assertEqual([decode(i).shape[1] for i in strip], [160, 160])
        with self.assertRaisesRegex(ToolError, 'past the end'):
            asyncio.run(server.preview_frame(spec, time_s=60))

    def test_preview_matches_full_render_for_history_effects(self):
        # Feedback carries history and the motor warps frame selection; a warmed-up
        # preview must still equal the rendered frame, and a cold one must not.
        from vhs_broadcast import api
        from vhs_broadcast.core.pipeline import FileSource, render_frames
        cases = {'feedback': {'feedback.enabled': 1},
                 'motor': {'feedback.enabled': 1, 'motor.enabled': 1, 'motor.depth': .5}}
        for name, parameters in cases.items():
            with self.subTest(name):
                take = str(self.dir / f'{name}.json')
                server.create_take({'parameters': dict(parameters, **{'text.enabled': 0})}, take)
                params, effects, timeline, source = api.load_take(take, self.clip)
                src = FileSource(source, read_ahead=False)
                try:
                    rendered = [f.copy() for _, f in zip(range(36), render_frames(
                        src, params, timeline, effects, max_width=160))][35]
                finally:
                    src.release()
                warm = server._preview(take, self.clip, [35 / 30], 160, 1.0)[0]
                np.testing.assert_array_equal(warm, rendered)
                cold = server._preview(take, self.clip, [35 / 30], 160, 0)[0]
                self.assertFalse(np.array_equal(cold, rendered))

    def test_bad_spec_is_a_clean_tool_error(self):
        with self.assertRaises(ToolError) as caught:
            server.create_take({'parameters': {'made.up': 1}}, str(self.dir / 'bad.json'))
        self.assertEqual(str(caught.exception), 'Unknown parameter: made.up')
        self.assertFalse((self.dir / 'bad.json').exists())

        async def call():
            async with Client(server.mcp) as client:
                return await client.call_tool('create_take', {
                    'spec': {'parameters': {'block_corrupt.density': 2}},
                    'output_path': str(self.dir / 'bad.json')})
        result = asyncio.run(call())
        self.assertTrue(result.is_error)
        text = result.content[0].text
        self.assertIn('block_corrupt.density: value must be in [0.0, 1.0]', text)
        self.assertNotIn('Traceback', text)

    def test_render_over_the_protocol_reports_progress(self):
        take = str(self.dir / 'render-take.json')
        server.create_take({'parameters': {'text.enabled': 0, 'block_corrupt.enabled': 1}}, take)
        output = str(self.dir / 'out.mp4')
        progress = []

        async def call():
            async def on_progress(value, total, message):
                progress.append(value)
            async with Client(server.mcp) as client:
                preview = await client.call_tool('preview_frame', {
                    'spec_or_take': take, 'source_video_path': self.clip,
                    'time_s': .5, 'width': 64})
                rendered = await client.call_tool('render', {
                    'take_path': take, 'output_path': output, 'source_video_path': self.clip,
                    'full': False, 'with_audio': False}, progress_callback=on_progress)
                return preview, rendered
        preview, rendered = asyncio.run(call())
        self.assertFalse(preview.is_error, preview.content)
        self.assertEqual(preview.content[0].type, 'image')
        self.assertEqual(decode(base64.b64decode(preview.content[0].data)).shape[1], 64)
        self.assertFalse(rendered.is_error, rendered.content)
        self.assertEqual(json.loads(rendered.content[0].text)['path'], str(Path(output).resolve()))
        self.assertGreater(cv2.VideoCapture(output).get(cv2.CAP_PROP_FRAME_COUNT), 0)
        self.assertTrue(progress)


if __name__ == '__main__':
    unittest.main()
