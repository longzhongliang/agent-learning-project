import json
import unittest

from app.tools.execution import ToolExecutionResult


class ExecutionTests(unittest.TestCase):
    def test_tool_execution_result_serializes_retry_contract(self) -> None:
        result = ToolExecutionResult(
            ok=False,
            message="Python 文件运行失败。",
            error_type="process_exit_nonzero",
            retry_strategy="after_state_verification",
            side_effect_status="unknown",
            exit_code=1,
            stdout="",
            stderr="boom",
        )

        payload = json.loads(result.model_dump_json())

        self.assertFalse(payload["ok"])
        self.assertEqual(payload["retry_strategy"], "after_state_verification")
        self.assertEqual(payload["side_effect_status"], "unknown")
