import importlib.util
from pathlib import Path
import unittest


def _load_bridge_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run" / "noitom_to_udp_bvh_bridge.py"
    spec = importlib.util.spec_from_file_location("noitom_to_udp_bvh_bridge", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NoitomUdpBridgeTests(unittest.TestCase):
    def test_extracts_floats_from_axis_text_payload(self):
        bridge = _load_bridge_module()

        values = bridge.extract_floats(b"Frame 42\n1 -2.5 3.25e1 ignored")

        self.assertEqual(values, [42.0, 1.0, -2.5, 32.5])

    def test_normalizes_rotation_only_frame_by_adding_zero_root_translation(self):
        bridge = _load_bridge_module()
        values = [float(i) for i in range(156)]

        frame = bridge.normalize_frame(values, expected_floats=159)

        self.assertEqual(frame[:3], [0.0, 0.0, 0.0])
        self.assertEqual(frame[3:], values)

    def test_preserves_sample_noitom_180_float_frame_by_default(self):
        bridge = _load_bridge_module()
        values = [float(i) for i in range(180)]

        frame = bridge.normalize_frame(values)

        self.assertEqual(frame, values)

    def test_noitom_udp_provider_registers_180_float_skeleton(self):
        from deploy.inputs.udp_bvh_provider import UDPBVHInputProvider

        provider = UDPBVHInputProvider(bvh_format="noitom", udp_port=0)
        try:
            self.assertEqual(provider.human_format, "noitom")
            self.assertEqual(provider._expected_floats, 180)
            self.assertIn("Hips", provider.bone_names)
            self.assertIn("LeftFoot", provider.bone_names)
            self.assertIn("RightFoot", provider.bone_names)
            self.assertIn("Neck1", provider.bone_names)
            self.assertIn("LeftInHandIndex", provider.bone_names)
            self.assertEqual(len(provider.bone_names), 59)
        finally:
            provider.close()

    def test_hc_mocap_udp_format_is_removed(self):
        from deploy.inputs.udp_bvh_provider import UDPBVHInputProvider

        with self.assertRaisesRegex(ValueError, "Unsupported bvh_format 'hc_mocap'"):
            UDPBVHInputProvider(bvh_format="hc_mocap", udp_port=0)

    def test_udp_provider_defaults_to_noitom(self):
        from deploy.inputs.udp_bvh_provider import UDPBVHInputProvider

        provider = UDPBVHInputProvider(udp_port=0)
        try:
            self.assertEqual(provider.human_format, "noitom")
            self.assertEqual(provider._expected_floats, 180)
        finally:
            provider.close()

    def test_hc_mocap_retarget_config_is_removed(self):
        from deploy.retargeting.gmr.params import IK_CONFIG_DICT

        self.assertNotIn("bvh_hc_mocap", IK_CONFIG_DICT)

    def test_sample_run_bvh_first_frame_matches_bridge_default_length(self):
        bridge = _load_bridge_module()
        lines = Path(__file__).resolve().parents[1].joinpath("data/sample_bvh/run.bvh").read_text().splitlines()
        motion_index = lines.index("MOTION")
        values = [float(value) for value in lines[motion_index + 3].split()]

        frame = bridge.normalize_frame(values)

        self.assertEqual(len(values), 180)
        self.assertEqual(len(frame), 180)
        self.assertEqual(frame, values)

    def test_noitom_provider_accepts_sample_run_bvh_frame(self):
        from deploy.inputs.udp_bvh_provider import UDPBVHInputProvider

        lines = Path(__file__).resolve().parents[1].joinpath("data/sample_bvh/run.bvh").read_text().splitlines()
        motion_index = lines.index("MOTION")
        payload = lines[motion_index + 3].encode("utf-8")

        provider = UDPBVHInputProvider(bvh_format="noitom", udp_port=0)
        try:
            provider._process_packet(payload)
            frame = provider.get_frame()
            self.assertIn("Hips", frame)
            self.assertIn("LeftFootMod", frame)
            self.assertIn("RightFootMod", frame)
        finally:
            provider.close()


if __name__ == "__main__":
    unittest.main()
