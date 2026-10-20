import ast
import os
import platform
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock


sourcePath = Path(__file__).parents[1] / "PPOCRLabel.py"
tree = ast.parse(sourcePath.read_text(encoding="utf-8"))
moveFunction = next(
    node
    for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "moveFileToTrash"
)
namespace = {
    "os": os,
    "platform": platform,
    "subprocess": subprocess,
    "logger": type("Logger", (), {"debug": staticmethod(lambda *_args: None)})(),
}
exec(
    compile(ast.Module([moveFunction], type_ignores=[]), str(sourcePath), "exec"),
    namespace,
)


class WindowsTrashTest(unittest.TestCase):
    def test_failed_delete_keeps_annotations_and_success_cleans_all_records(self):
        delete_function = next(
            n
            for c in tree.body
            if isinstance(c, ast.ClassDef) and c.name == "MainWindow"
            for n in c.body
            if isinstance(n, ast.FunctionDef) and n.name == "deleteImg"
        )
        for succeeded in (False, True):
            with self.subTest(succeeded=succeeded):
                window = Mock()
                window.filePath = r"C:\dataset\images\sample.png"
                window.lastOpenDir = r"C:\dataset\images"
                window.deleteImgDialog.return_value = 1
                window.getImglabelidx.return_value = "images/sample.png"
                window.fileStatedict = {"images/sample.png": 1}
                window.PPlabel = {"images/sample.png": [{"transcription": "A0514+"}]}
                window.Cachelabel = dict(window.PPlabel)
                scope = {
                    "QMessageBox": Mock(Yes=1),
                    "moveFileToTrash": Mock(return_value=succeeded),
                    "logger": Mock(),
                }
                exec(
                    compile(
                        ast.Module([delete_function], type_ignores=[]),
                        str(sourcePath),
                        "exec",
                    ),
                    scope,
                )
                scope["deleteImg"](window)
                self.assertEqual(bool(window.fileStatedict), not succeeded)
                self.assertEqual(bool(window.PPlabel), not succeeded)
                self.assertEqual(bool(window.Cachelabel), not succeeded)
                if succeeded:
                    window.saveCacheLabel.assert_called_once()
                    window.importDirImages.assert_called_once_with(
                        window.lastOpenDir, isDelete=True
                    )
                else:
                    window.saveCacheLabel.assert_not_called()
                    window.importDirImages.assert_not_called()

    @unittest.skipUnless(platform.system() == "Windows", "Windows-only check")
    def test_explicit_path_buffer(self):
        with tempfile.NamedTemporaryFile(
            prefix="ppocrlabel_trash_", delete=False
        ) as file:
            tempPath = Path(file.name)

        try:
            self.assertTrue(namespace["moveFileToTrash"](str(tempPath)))
            self.assertFalse(tempPath.exists())
        finally:
            tempPath.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
