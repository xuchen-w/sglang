"""Serve hybrid GDN Qwen3.5 through the compiled MLX runner and match eager MPS."""

import importlib.util
import os
import re
import tempfile
import unittest

import requests
import torch

from sglang.test.ci.ci_register import register_mps_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
    try_cached_model,
)

register_mps_ci(est_time=600, suite="extra-e2e-test-mps")

MODEL = "Qwen/Qwen3.5-0.8B"

PROMPTS = [
    "The capital of France is",
    "The first five prime numbers are",
    "Write a haiku about the ocean:",
    # Longer than one 64-token prefill chunk, so the GDN state crosses chunks.
    "Explain the theory of relativity in simple terms. " * 3,
]


@unittest.skipUnless(
    torch.backends.mps.is_available() and importlib.util.find_spec("mlx") is not None,
    "Requires Torch MPS and MLX",
)
class TestQwen35CompiledMlx(CustomTestCase):
    def _generate(self, payload):
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate", json=payload, timeout=600
        )
        response.raise_for_status()
        return response.json()

    def _greedy(self, text, tokens=16):
        return self._generate(
            {
                "text": text,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": tokens,
                    "ignore_eos": True,
                },
            }
        )

    def _launch(self, backend, dtype, log, extra=()):
        return popen_launch_server(
            try_cached_model(MODEL),
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env=dict(os.environ, SGLANG_USE_MLX="0"),
            device="mps",
            return_stdout_stderr=(log, log),
            other_args=[
                "--device",
                "mps",
                "--json-model-override-args",
                '{"language_model_only":true}',
                "--dtype",
                dtype,
                "--mps-execution-backend",
                backend,
                "--context-length",
                "1024",
                "--max-total-tokens",
                "4096",
                "--max-running-requests",
                "4",
                "--mem-fraction-static",
                "0.5",
                *extra,
                # The prefill execution line is logged at debug level.
                *(("--log-level", "debug") if backend == "mlx-compiled" else ()),
            ],
        )

    def test_greedy_matches_eager_and_uses_every_compiled_path(self):
        results = {}
        logs = {}
        for backend, dtype in (("eager", "bfloat16"), ("mlx-compiled", "bfloat16")):
            log = self.enterContext(tempfile.TemporaryFile(mode="w+"))
            process = self._launch(backend, dtype, log)
            try:
                results[backend] = [self._greedy(p)["output_ids"] for p in PROMPTS]
                # A longer prompt first runs a compiled batch-1 prefill, then a
                # repeat reuses the cached prefix and must decode the same.
                long_ids = [42, 43, 44] * 45 + [42]
                first = self._generate(
                    {
                        "input_ids": long_ids,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 8,
                            "ignore_eos": True,
                        },
                    }
                )
                self.assertEqual(first["meta_info"]["cached_tokens"], 0)
                warm = self._generate(
                    {
                        "input_ids": long_ids + first["output_ids"] + [43, 44],
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 8,
                            "ignore_eos": True,
                        },
                    }
                )
                self.assertGreater(warm["meta_info"]["cached_tokens"], len(long_ids))
                requests.post(DEFAULT_URL_FOR_TEST + "/flush_cache", timeout=60)
                cold = self._generate(
                    {
                        "input_ids": long_ids + first["output_ids"] + [43, 44],
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 8,
                            "ignore_eos": True,
                        },
                    }
                )
                self.assertEqual(cold["meta_info"]["cached_tokens"], 0)
                self.assertEqual(cold["output_ids"], warm["output_ids"])
                # Long enough for the runner to report its lookahead counters.
                self._greedy(PROMPTS[2], 128)
            finally:
                terminate_and_kill_process_tree(process)
            log.seek(0)
            logs[backend] = log.read()
        for index, prompt in enumerate(PROMPTS):
            self.assertEqual(
                results["eager"][index],
                results["mlx-compiled"][index],
                prompt,
            )
        compiled = logs["mlx-compiled"]
        self.assertIn("gdn_layers=18", compiled)
        # The first long prompt and the post-flush rerun of it plus its output
        # are both compiled batch-1 prefills.
        self.assertIn("Compiled MLX prefill execution: tokens=136", compiled)
        self.assertIn("Compiled MLX prefill execution: tokens=146", compiled)
        # 1-token prefills use a static graph; all longer ones share one dynamic graph.
        sizes = map(int, re.findall(r"prefill graph ready: size=(\d+)", compiled))
        dynamic_graphs = [size for size in sizes if size >= 2]
        self.assertEqual(len(dynamic_graphs), 1, dynamic_graphs)
        self.assertRegex(compiled, r"Compiled MLX: .*executions=\d+, enqueued=[1-9]\d*")
        self.assertRegex(compiled, r"Compiled MLX: .*hits=[1-9]\d*")
        self.assertIn(
            "using eager MPS: prefill requires an uncached B1 prompt", compiled
        )
        self.assertNotIn("using eager MPS: batch outside", compiled)
        self.assertNotIn("using eager MPS: state tracking", compiled)

    def test_fp32_greedy_matches_the_eager_fp32_run_exactly(self):
        """/tmp reference: the compiled graph must not bend fp32 greedy tokens."""
        results = {}
        logs = {}
        for backend in ("eager", "mlx-compiled"):
            log = self.enterContext(tempfile.TemporaryFile(mode="w+"))
            process = self._launch(
                backend, "float32", log, ("--chunked-prefill-size", "64")
            )
            try:
                results[backend] = [self._greedy(p, 24)["output_ids"] for p in PROMPTS]
            finally:
                terminate_and_kill_process_tree(process)
            log.seek(0)
            logs[backend] = log.read()
        for index, prompt in enumerate(PROMPTS):
            self.assertEqual(
                results["eager"][index], results["mlx-compiled"][index], prompt
            )
        compiled = logs["mlx-compiled"]
        # Only the decode runner enqueues lookahead.
        self.assertRegex(compiled, r"Compiled MLX: .*enqueued=[1-9]")
        self.assertIn("gdn_layers=18", compiled)
        # A fallback logs its reason the first time; only prefill may fall back.
        unexpected = [
            reason
            for reason in re.findall(r"using eager MPS: (.*)", compiled)
            if not reason.startswith("prefill requires an uncached B1 prompt")
        ]
        self.assertEqual(unexpected, [])


if __name__ == "__main__":
    unittest.main()
