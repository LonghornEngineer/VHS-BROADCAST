import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

import cv2
import numpy as np

from vhs_broadcast.api import catalog, create_take, validate_take, main
from vhs_broadcast.core.automation import Session, Timeline
from vhs_broadcast.core.parameters import ParameterSet
from vhs_broadcast.effects.vhs import default_effects


class AgentApiTests(unittest.TestCase):
    def call(self, args):
        stream = io.StringIO()
        with redirect_stdout(stream):
            code = main(args)
        return code, json.loads(stream.getvalue())

    def test_catalog_complete_and_json_command(self):
        result = catalog()
        ids = [p['id'] for e in result['effects'] for p in e['parameters']]
        params = ParameterSet()
        default_effects(params)
        self.assertEqual(set(ids), {p.id for p in params.all()})
        self.assertEqual(len(ids), len(set(ids)))
        code, data = self.call(['catalog'])
        self.assertEqual(code, 0)
        self.assertTrue(data['ok'])
        self.assertEqual(data['result']['api_version'], 1)

    def test_documented_example_opens_as_native_take(self):
        example = Path(__file__).resolve().parents[1] / 'examples/block-glitch.json'
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / 'take.json')
            code, data = self.call(['create', '--spec', str(example), '--output', output])
            self.assertEqual(code, 0, data)
            report = validate_take(output)
            self.assertEqual(report['tracks'], 1)
            self.assertTrue(report['warnings'])
            p = ParameterSet(); e = default_effects(p); t = Timeline()
            Session(p, t, e).load(output)
            self.assertEqual(p.get('block_corrupt.mode').value, 3)
            self.assertEqual(t.tracks['block_corrupt.density'].values, [0, .1, .65, 0])

    def test_invalid_values_keys_modulation_and_settings_never_write(self):
        cases = [
            {'parameters': {'made.up': 1}},
            {'parameters': {'block_corrupt.density': 2}},
            {'parameters': {'block_corrupt.density': float('nan')}},
            {'parameters': {'block_corrupt.mode': 1.5}},
            {'parameters': {'block_corrupt.density': {'value': .2, 'mod': {'enabled': 'false'}}}},
            {'parameters': {'block_corrupt.density': {'value': .2, 'audio': {'band': 'oops'}}}},
            {'timeline': {'tracks': {'block_corrupt.density': {'t': [0, 1], 'v': [.1]}}}},
            {'timeline': {'tracks': {'block_corrupt.density': {'t': [1, 0], 'v': [.1, .2]}}}},
            {'timeline': {'tracks': {'block_corrupt.mode': {'t': [0], 'v': [1], 'interpolation': 'linear'}}}},
            {'effect_settings': {'text': {'manual_datetime': 'not a time'}}},
            {'order': ['text', 'text']}, {'typo': 123},
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'take.json'
            for case in cases:
                with self.subTest(case=case), self.assertRaises((ValueError, TypeError)):
                    create_take(case, str(output))
                self.assertFalse(output.exists())

    def test_overwrite_protection_and_stdin(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'take.json'
            output.write_text('existing', encoding='utf-8')
            with self.assertRaises(ValueError):
                create_take({}, str(output))
            self.assertEqual(output.read_text(), 'existing')
            with patch('sys.stdin', io.StringIO('{"parameters":{"text.enabled":0}}')):
                code, result = self.call(['create', '--spec', '-', '--output', str(output), '--overwrite'])
            self.assertEqual(code, 0, result)
            self.assertTrue(validate_take(str(output))['valid'])
            with self.assertRaises(ValueError):
                create_take({'source': str(output)}, str(output), overwrite=True)

    def test_failures_return_parseable_json(self):
        code, result = self.call(['validate', '--session', 'nonexistent-take-for-test.json'])
        self.assertEqual(code, 1)
        self.assertFalse(result['ok'])
        self.assertIn('error', result)

    @patch('vhs_broadcast.core.pipeline._AvFileSource._detect_hw', return_value=None)
    def test_agent_render_produces_video_and_protects_destination(self, _probe):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, take, output = map(str, (root/'source.avi', root/'take.json', root/'out.mp4'))
            writer = cv2.VideoWriter(source, cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 48))
            for i in range(4):
                writer.write(np.full((48, 64, 3), 50+i*40, np.uint8))
            writer.release()
            create_take({'source': source, 'parameters': {'text.enabled': 0, 'palette.enabled': 1}}, take)
            args = ['render', '--session', take, '--output', output, '--width', '32', '--no-audio']
            code, result = self.call(args)
            self.assertEqual(code, 0, result)
            capture = cv2.VideoCapture(output)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 4)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 32)
            capture.release()
            self.assertEqual(self.call(args)[0], 1)
            self.assertEqual(self.call(args + ['--overwrite'])[0], 0)
            create_take({'source': source, 'parameters': {
                'text.enabled': 0, 'downsample.enabled': 1,
                'downsample.amount': 0, 'downsample.fps': 25}}, take, overwrite=True)
            self.assertEqual(self.call(args + ['--overwrite'])[0], 0)
            capture = cv2.VideoCapture(output)
            frames = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
            capture.release()
            self.assertEqual(len(frames), 4)
            for frame in frames[1:]:
                np.testing.assert_allclose(frame.astype(float), frames[0].astype(float), atol=3)
