"""Verify mask boundaries, credential failures and real MCP input preparation."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

SOURCE = Path(__file__).resolve().parents[1] / "tools/kimi_image_tools.py"
sys.path.insert(0, str(SOURCE.parent))
spec = importlib.util.spec_from_file_location("kimi_image_tools_tested", SOURCE)
tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tools)


class KimiImageToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.scope = patch.object(tools, "ROOT", self.root)
        self.scope.start()
        self.source = self.root / "source.png"
        self.mask = self.root / "selection.png"
        self.pixels = np.zeros((24, 32, 3), dtype=np.uint8)
        self.pixels[..., 2] = 180
        Image.fromarray(self.pixels).save(self.source)
        selected = np.zeros((24, 32), dtype=np.uint8)
        selected[7:13, 9:19] = 255
        Image.fromarray(selected).save(self.mask)
        self.selected = selected > 0

    def tearDown(self):
        self.scope.stop()
        self.temp.cleanup()

    def test_alpha_mask_marks_only_selection_transparent(self):
        image = Image.open(self.source)
        selection = Image.open(self.mask)
        canvas, mask, box = tools.prepare_canvas(image, selection)
        alpha = np.asarray(mask.getchannel("A"))
        self.assertEqual(alpha[0, 0], 255)
        self.assertEqual(alpha[400, 600], 0)
        self.assertEqual(canvas.size, mask.size)
        self.assertLess(box[1], box[3])

    def test_empty_selection_does_not_call_service(self):
        Image.new("L", (32, 24), 0).save(self.mask)
        with patch.object(tools.subprocess, "run") as command:
            result = tools.rebuild_background(str(self.source), str(self.mask), dry_run=False)
        self.assertEqual(result["status"], "empty_selection")
        command.assert_not_called()

    def test_rejects_wrong_mask_dimensions_and_transparent_layer(self):
        Image.new("L", (2, 2), 255).save(self.mask)
        with self.assertRaisesRegex(ValueError, "尺寸"):
            tools.rebuild_background(str(self.source), str(self.mask))
        Image.new("RGBA", (32, 24), (255, 0, 0, 90)).save(self.mask)
        with self.assertRaisesRegex(ValueError, "黑白"):
            tools.rebuild_background(str(self.source), str(self.mask))

    def test_missing_image_key_fails_before_upload(self):
        with patch.object(tools, "api_key", return_value=""), \
                patch.object(tools.subprocess, "run") as command:
            with self.assertRaisesRegex(ValueError, "尚未配置"):
                tools.rebuild_background(str(self.source), str(self.mask), dry_run=False)
        command.assert_not_called()

    def test_final_composite_keeps_unselected_pixels_exactly(self):
        def generate(command, **kwargs):
            input_path = command[command.index("--image") + 1]
            output_path = command[command.index("--out") + 1]
            with Image.open(input_path) as source:
                Image.new("RGB", source.size, (220, 50, 10)).save(output_path)
            self.assertEqual(kwargs["env"]["OPENAI_API_KEY"], "sk-test-secret")
            self.assertNotIn("OPENAI_BASE_URL", kwargs["env"])
            return subprocess.CompletedProcess(command, 0, "generated", "")

        with patch.object(tools, "api_key", return_value="sk-test-secret"), \
                patch.object(tools.subprocess, "run", side_effect=generate):
            result = tools.rebuild_background(str(self.source), str(self.mask), dry_run=False)
        output = np.asarray(Image.open(result["result"]))
        self.assertTrue(np.array_equal(output[~self.selected], self.pixels[~self.selected]))
        # A flat generated color should match the original boundary, not leave red letters.
        self.assertTrue(np.all(output[self.selected] == [0, 0, 180]))
        self.assertEqual(result["composition_method"], "generated_gradients_with_original_boundary")
        self.assertFalse(result["quality_verified"])
        self.assertTrue(result["outside_selection_unchanged"])
        self.assertTrue(np.array_equal(np.asarray(Image.open(self.source)), self.pixels))

    def test_error_log_redacts_credentials(self):
        with patch.object(tools, "api_key", return_value="sk-test-secret"), \
                patch.object(tools.subprocess, "run", return_value=(
                    subprocess.CompletedProcess([], 1, "", "rejected sk-test-secret"))):
            result = tools.rebuild_background(str(self.source), str(self.mask), dry_run=False)
        log = (Path(result["run_directory"]) / "imagegen.log").read_text()
        self.assertNotIn("sk-test-secret", log)
        self.assertEqual(result["status"], "error")
        self.assertNotIn("result", result)

    def test_abnormal_output_dimensions_are_not_silently_composited(self):
        def generate(command, **kwargs):
            Image.new("RGB", (2, 2), "red").save(command[command.index("--out") + 1])
            return subprocess.CompletedProcess(command, 0, "", "")
        with patch.object(tools, "api_key", return_value="sk-test-secret"), \
                patch.object(tools.subprocess, "run", side_effect=generate):
            result = tools.rebuild_background(str(self.source), str(self.mask), dry_run=False)
        self.assertEqual(result["status"], "size_mismatch")
        self.assertNotIn("result", result)

    def test_symlink_to_outside_project_cannot_be_uploaded(self):
        link = self.root / "outside.png"
        link.symlink_to(SOURCE)
        with self.assertRaisesRegex(ValueError, "项目目录"):
            tools.project_file(str(link))

    def test_bundled_cli_dry_run_prepares_edit_without_api_call(self):
        result = tools.rebuild_background(str(self.source), str(self.mask))
        self.assertEqual(result["status"], "dry_run")
        self.assertFalse(result["api_called"])
        run = Path(result["run_directory"])
        self.assertTrue((run / "model_mask.png").is_file())
        self.assertFalse((run / "generated.png").exists())
        self.assertEqual(json.loads((run / "manifest.json").read_text())["model"], tools.MODEL)


if __name__ == "__main__":
    unittest.main()
