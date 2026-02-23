import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sow.v2.baseline_inference import (  # noqa: E402
    NonFiniteBatchError,
    execute_with_batch_backoff,
)


class TestStage00aDynamicBatchBackoff(unittest.TestCase):
    def test_oom_backoff_retries_same_offset_and_succeeds(self) -> None:
        items = list(range(9))
        processed = []

        def _run_batch(batch):
            if len(batch) > 2:
                raise RuntimeError("out of memory")
            processed.extend(batch)

        result = execute_with_batch_backoff(
            items=items,
            batch_chain=[6, 5, 4, 3, 2, 1],
            run_batch=_run_batch,
        )
        self.assertEqual(processed, items)
        self.assertTrue(all(int(x) <= 2 for x in (result.get("batch_sizes_used") or [])))

    def test_non_finite_backoff_uses_smaller_batch(self) -> None:
        items = list(range(5))
        calls = []

        def _run_batch(batch):
            calls.append(len(batch))
            if len(batch) > 1:
                raise NonFiniteBatchError("non-finite activation")

        result = execute_with_batch_backoff(
            items=items,
            batch_chain=[3, 2, 1],
            run_batch=_run_batch,
        )
        self.assertIn(1, calls)
        self.assertGreaterEqual(int(result.get("rows_processed", 0)), len(items))

    def test_backoff_fails_closed_when_batch_size_one_fails(self) -> None:
        items = [1, 2, 3]

        def _run_batch(_batch):
            raise RuntimeError("out of memory")

        with self.assertRaises(RuntimeError):
            execute_with_batch_backoff(
                items=items,
                batch_chain=[2, 1],
                run_batch=_run_batch,
            )


if __name__ == "__main__":
    unittest.main()
