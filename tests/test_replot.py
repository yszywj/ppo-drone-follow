from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from pegasus_iris_fast_line_follow.replot_pegasus_iris_fast_line_follow import (
    load_metrics_rows,
    parse_csv_value,
)


class ReplotMetricsLoaderTest(unittest.TestCase):
    def test_scalar_types_are_restored_from_csv(self) -> None:
        self.assertIs(parse_csv_value("True"), True)
        self.assertIs(parse_csv_value("False"), False)
        self.assertEqual(parse_csv_value("12"), 12)
        self.assertAlmostEqual(parse_csv_value("0.25"), 0.25)
        self.assertEqual(parse_csv_value("success"), "success")

    def test_metrics_csv_is_loaded_as_typed_rows(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=("total_steps", "success", "rate"),
                )
                writer.writeheader()
                writer.writerow(
                    {"total_steps": 64, "success": False, "rate": 0.75}
                )
            self.assertEqual(
                load_metrics_rows(path),
                [{"total_steps": 64, "success": False, "rate": 0.75}],
            )


if __name__ == "__main__":
    unittest.main()
