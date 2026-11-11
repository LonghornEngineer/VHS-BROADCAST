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
        effects = default_effects(params)
        retired = {p.id for e in effects for p in [e.enabled_param(), *e.own_params()]
                   if e.legacy_of or p.legacy}
        self.assertEqual(set(ids), {p.id for p in params.all()} - retired)
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


class ExampleTakeTests(unittest.TestCase):
    def test_every_example_opens_as_a_take(self):
        """The GUI's Open Take and the render CLI accept the spec shorthand the examples use."""
        from vhs_broadcast.core.automation import Session, Timeline
        from vhs_broadcast.core.parameters import ParameterSet
        from vhs_broadcast.effects.vhs import default_effects
        examples = sorted((Path(__file__).resolve().parents[1] / "examples").glob("*.json"))
        self.assertTrue(examples)
        for path in examples:
            with self.subTest(example=path.name):
                params = ParameterSet()
                effects = default_effects(params)
                session = Session(params, Timeline(), effects)
                session.load(str(path))
                spec = json.loads(path.read_text(encoding="utf-8"))
                for pid, value in spec["parameters"].items():
                    if isinstance(value, (int, float)):
                        self.assertEqual(params.get(pid).value, params.get(pid).clamp(value))


class HeldKeyWarningTests(unittest.TestCase):
    def spec(self, retrace=None, enabled=None):
        tracks = {}
        if retrace:
            tracks["circuit_bend.retrace"] = {"t": [t for t, _ in retrace], "v": [v for _, v in retrace],
                                              "interpolation": "linear"}
        if enabled:
            tracks["circuit_bend.enabled"] = {"t": [t for t, _ in enabled], "v": [v for _, v in enabled],
                                              "interpolation": "hold"}
        return {"parameters": {"circuit_bend.enabled": 1}, "timeline": {"tracks": tracks}}

    def warnings(self, spec):
        with tempfile.TemporaryDirectory() as d:
            result = create_take(spec, str(Path(d) / "take.json"))
            self.assertEqual(result["warnings"], validate_take(result["path"])["warnings"][1:])
            return result["warnings"]

    def test_a_late_first_key_while_the_effect_is_on_is_reported(self):
        found = self.warnings(self.spec(retrace=[(10, .6), (12, 0)]))
        self.assertEqual(len(found), 1)
        self.assertIn("circuit_bend.retrace", found[0])
        self.assertIn("0:10.00", found[0])
        self.assertIn("on from 0:00.00", found[0])

    def test_no_warning_when_the_take_is_unambiguous(self):
        cases = {
            "keyed from the start": self.spec(retrace=[(0, 0), (10, .6)]),
            "first value is the default": self.spec(retrace=[(10, 0), (12, .6)]),
            "effect only turns on at the key": self.spec(retrace=[(10, .6)], enabled=[(0, 0), (10, 1)]),
            "on less than a frame early": self.spec(retrace=[(10.01, .6)], enabled=[(0, 0), (10, 1)]),
        }
        for name, spec in cases.items():
            with self.subTest(name):
                self.assertEqual(self.warnings(spec), [])

    def test_effect_switched_on_before_the_key_by_its_own_track(self):
        found = self.warnings(self.spec(retrace=[(10, .6)], enabled=[(0, 0), (4, 1)]))
        self.assertEqual(len(found), 1)
        self.assertIn("on from 0:04.00", found[0])
