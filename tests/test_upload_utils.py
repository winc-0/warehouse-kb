# tests/test_upload_utils.py
"""上传文件名校验：保存路径不能被客户端填写的文件名带出上传目录。"""

import os
import unittest

from utils.upload_utils import UnsafeFilenameError, safe_upload_name


class TestSafeUploadName(unittest.TestCase):
    def test_普通文件名原样保留(self):
        self.assertEqual(safe_upload_name("CPD30叉车手册.pdf"), "CPD30叉车手册.pdf")
        self.assertEqual(safe_upload_name("  manual v2.md "), "manual v2.md")

    def test_只保留最后一段(self):
        for raw in ("../../evil.md", "..\\..\\evil.md", "/etc/evil.md", "a/b/evil.md"):
            with self.subTest(raw=raw):
                self.assertEqual(safe_upload_name(raw), "evil.md")

    def test_拼接后仍在上传目录内(self):
        base = os.path.join("data", "20261106", "task")
        for raw in ("../../evil.md", "C:/Windows/Temp/evil.md", "\\\\server\\share\\evil.md"):
            with self.subTest(raw=raw):
                path = os.path.join(base, safe_upload_name(raw))
                self.assertEqual(os.path.dirname(path), base)

    def test_拒绝无效名字(self):
        for raw in (None, "", "   ", ".", "..", "dir/", "C:evil.md", "evil\x00.md"):
            with self.subTest(raw=raw):
                with self.assertRaises(UnsafeFilenameError):
                    safe_upload_name(raw)


if __name__ == "__main__":
    unittest.main()
