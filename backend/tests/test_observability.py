import threading
import unittest

from src.observability.metrics import MetricsRegistry


class MetricsRegistryTests(unittest.TestCase):
    def test_counter_accumulates_by_labels(self):
        reg = MetricsRegistry()
        reg.inc("physioai_test_total", method="GET")
        reg.inc("physioai_test_total", 2, method="GET")
        snap = reg.snapshot()
        row = next(x for x in snap["counters"] if x["name"] == "physioai_test_total")
        self.assertEqual(row["value"], 3)
        self.assertEqual(row["labels"], {"method": "GET"})

    def test_gauge_increment_and_decrement(self):
        reg = MetricsRegistry()
        reg.inc_gauge("physioai_active", 2)
        reg.inc_gauge("physioai_active", -1)
        self.assertEqual(reg.get_gauge("physioai_active"), 1)

    def test_timer_summary(self):
        reg = MetricsRegistry()
        reg.observe("physioai_duration", 0.1, route="/health")
        reg.observe("physioai_duration", 0.3, route="/health")
        row = next(x for x in reg.snapshot()["timers"] if x["name"] == "physioai_duration")
        self.assertEqual(row["count"], 2)
        self.assertAlmostEqual(row["sum_seconds"], 0.4)
        self.assertAlmostEqual(row["avg_seconds"], 0.2)
        self.assertAlmostEqual(row["max_seconds"], 0.3)

    def test_prometheus_export_escapes_labels(self):
        reg = MetricsRegistry()
        reg.inc("physioai_events_total", kind='a"b')
        text = reg.to_prometheus()
        self.assertIn('physioai_events_total{kind="a\\"b"} 1.000000', text)

    def test_registry_is_thread_safe(self):
        reg = MetricsRegistry()
        def worker():
            for _ in range(1000):
                reg.inc("physioai_thread_total")
        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        row = next(x for x in reg.snapshot()["counters"] if x["name"] == "physioai_thread_total")
        self.assertEqual(row["value"], 4000)


if __name__ == "__main__":
    unittest.main()
