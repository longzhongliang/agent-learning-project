from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from app.tools.filesystem import (
    WorkspacePathError,
    read_text_file,
    resolve_workspace_path,
    search_text_in_file,
    write_text_file,
)


class FilesystemTests(unittest.TestCase):
    def test_resolve_workspace_path_accepts_relative_path(self) -> None:
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            self.assertEqual(
                resolve_workspace_path(workspace, "notes/today.txt"),
                (workspace / "notes" / "today.txt").resolve(),
            )

    def test_resolve_workspace_path_rejects_escape(self) -> None:
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            for requested in ("../outside.txt", "..\\outside.txt", "C:\\outside.txt"):
                with self.subTest(requested=requested):
                    with self.assertRaises(WorkspacePathError):
                        resolve_workspace_path(workspace, requested)

    def test_write_and_read_text_file(self) -> None:
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            result = write_text_file(workspace, "hello.txt", "你好，Agent")
            self.assertIn("写入成功", result)
            self.assertEqual(read_text_file(workspace, "hello.txt"), "你好，Agent")

    def test_write_requires_existing_parent_directory(self) -> None:
        with TemporaryDirectory() as directory:
            self.assertIn("父目录不存在", write_text_file(Path(directory), "missing/hello.txt", "content"))

    def test_search_text_returns_line_numbers(self) -> None:
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "example.txt").write_text("first\nneedle here\nneedle again", encoding="utf-8")
            self.assertEqual(
                search_text_in_file(workspace, "example.txt", "needle"),
                "2: needle here\n3: needle again",
            )
