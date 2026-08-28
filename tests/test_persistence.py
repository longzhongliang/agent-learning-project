from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from app.persistence import SessionRepository, create_store


class PersistenceTests(unittest.TestCase):
    def test_session_repository_creates_and_lists_sessions(self) -> None:
        with TemporaryDirectory() as directory:
            repository = SessionRepository(Path(directory) / "agent.db")
            try:
                repository.create_session("thread-1", "测试会话")
                sessions = repository.list_sessions()
                self.assertEqual(len(sessions), 1)
                self.assertEqual(sessions[0].thread_id, "thread-1")
                self.assertEqual(sessions[0].title, "测试会话")
            finally:
                repository.close()

    def test_store_persists_long_term_memory_across_contexts(self) -> None:
        with TemporaryDirectory() as directory:
            database_path = Path(directory) / "agent.db"

            with create_store(database_path) as store:
                store.put(("user_preferences",), "language", {"value": "zh-CN"})

            with create_store(database_path) as store:
                item = store.get(("user_preferences",), "language")

            self.assertIsNotNone(item)
            self.assertEqual(item.value, {"value": "zh-CN"})
