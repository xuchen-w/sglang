"""Live weights and dependent asynchronous launches for direct MLX export."""

import gc
import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_mps_ci
from sglang.test.test_utils import CustomTestCase

register_mps_ci(est_time=30, suite="extra-unit-test-mps")


@unittest.skipUnless(
    torch.backends.mps.is_available() and importlib.util.find_spec("mlx") is not None,
    "Requires Torch MPS and MLX",
)
class TestCompiledGraph(CustomTestCase):
    def test_explicit_pending_kv_matches_exported_mlx_without_committing(self):
        from sglang.kernels.ops.attention.mlx.radix_attention import radix_decode
        from sglang.kernels.ops.attention.mlx.radix_attention_export import (
            radix_decode as reference,
        )
        from sglang.srt.hardware_backend.mps.compiled_graph import CompiledMlxGraph
        from sglang.srt.utils.tensor_bridge import mlx_to_torch

        class Model(torch.nn.Module):
            def forward(self, q, k, v, pool, table, rows, lengths, tk, tv):
                return reference(
                    q,
                    k,
                    v,
                    pool,
                    pool,
                    table,
                    rows,
                    lengths,
                    0.125,
                    tail_k=tk,
                    tail_v=tv,
                )

        q = torch.zeros(1, 4, 64, device="mps")
        k = torch.ones(1, 2, 64, device="mps")
        pool = torch.full((16, 2, 64), float("nan"), device="mps")
        table = torch.full((1, 16), -1, device="mps", dtype=torch.int32)
        inputs = (
            q,
            k,
            k * 3,
            pool,
            table,
            torch.tensor([0], device="mps"),
            torch.tensor([2], device="mps"),
            k,
            k * 5,
        )
        model = Model()
        graph = CompiledMlxGraph(
            model=model, example_inputs=inputs, attention=radix_decode
        )
        self.addCleanup(graph.close)
        torch.mps.synchronize()
        arrays = graph.bind(inputs)
        for lookahead in (False, True):
            with self.subTest(lookahead=lookahead):
                tails = (
                    tuple((array * 2)[None] for array in arrays[-2:])
                    if lookahead
                    else None
                )
                expected_inputs = (
                    (*inputs[:-2], *(tensor * 2 for tensor in inputs[-2:]))
                    if lookahead
                    else inputs
                )
                actual = mlx_to_torch(graph.launch(arrays, tails=tails)[0])
                torch.testing.assert_close(
                    actual, model(*expected_inputs), rtol=0, atol=0
                )
        self.assertTrue(torch.isnan(pool).all().item())
        self.assertTrue((table == -1).all().item())

    def _runner_fixture(self, *, prefill=False, tokens=1):
        from sglang.srt.hardware_backend.mps.compiled_runner import CompiledMlxRunner
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        from sglang.srt.layers.radix_attention import RadixAttention
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.model_executor.forward_mode import ForwardMode

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(vocab_size=17)
                self.embed = torch.nn.Embedding(17, 64, device="mps")
                self.layers = torch.nn.ModuleList([RadixAttention(1, 64, 0.125, 1, 0)])

            def forward(self, input_ids, positions, forward_batch):
                hidden = self.embed(input_ids) + positions[:, None]
                hidden = self.layers[0](hidden, hidden, hidden, forward_batch)
                if forward_batch.forward_mode.is_extend():
                    hidden = hidden[-1:]
                return LogitsProcessorOutput(next_token_logits=hidden[:, :17])

        pools = tuple(torch.randn(16, 1, 64, device="mps") for _ in range(2))
        table = torch.arange(1, 9, device="mps", dtype=torch.int32)[None, :]

        def commit(layer, locations, keys, values):
            pools[0][locations] = keys
            pools[1][locations] = values

        runner = object.__new__(CompiledMlxRunner)
        runner.model_runner = SimpleNamespace(
            model=Model().eval(),
            model_config=SimpleNamespace(context_len=8),
            req_to_token_pool=SimpleNamespace(req_to_token=table),
            token_to_kv_pool=SimpleNamespace(
                page_size=1,
                get_kv_buffer=lambda _: pools,
                get_key_buffer=lambda _: pools[0],
                set_kv_buffer=commit,
            ),
        )
        runner._prefill = prefill
        runner._unsupported_reason = runner._region = None
        runner._layers = ()
        runner._linear_layers = ()
        runner._state = None
        runner._max_batch_size = 1
        runner._graphs = {}
        runner._unsupported_batches = {}
        runner._pool_identity = runner._model_structure = None
        runner._weight_sources = ()
        runner._fallback_reasons = set()
        runner.execution_count = 0
        runner.compile_seconds = 0.0
        runner._init_backend()
        runner._refresh_region()
        self.addCleanup(lambda: [graph.close() for graph in runner._graphs.values()])
        batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND if prefill else ForwardMode.DECODE,
            batch_size=1,
            input_ids=torch.ones(tokens, device="mps", dtype=torch.int64),
            positions=torch.arange(tokens, device="mps"),
            req_pool_indices=torch.tensor([0], device="mps"),
            seq_lens=torch.tensor([tokens], device="mps"),
            seq_lens_sum=tokens,
            out_cache_loc=torch.arange(1, tokens + 1, device="mps"),
            extend_num_tokens=tokens if prefill else None,
            extend_seq_lens=torch.tensor([tokens], device="mps") if prefill else None,
            extend_seq_lens_cpu=[tokens] if prefill else None,
            extend_prefix_lens=torch.tensor([0], device="mps") if prefill else None,
            extend_prefix_lens_cpu=[0] if prefill else None,
            extend_start_loc=torch.tensor([0], device="mps") if prefill else None,
        )
        return runner, batch, pools

    def test_prelaunch_export_and_lazy_trace_failures_fall_back_without_kv_writes(self):
        """A shape must fail admission before a live request can reach a bad trace."""
        import mlx.core as mx

        from sglang.srt.runtime_context import get_context

        def lazy_failure(*args, **kwargs):
            def trace(*flat):
                raise RuntimeError("unsupported test MLX trace")

            return trace

        for boundary, failure in (
            (
                "export",
                patch(
                    "torch.export.export",
                    side_effect=RuntimeError("unsupported export"),
                ),
            ),
            (
                "non-dynamic export",
                patch(
                    "torch.export.export",
                    side_effect=torch._dynamo.exc.UserError(
                        torch._dynamo.exc.UserErrorType.INVALID_INPUT,
                        "unsupported non-dynamic export",
                    ),
                ),
            ),
            ("lazy trace", patch.object(mx, "compile", side_effect=lazy_failure)),
            (
                "compile construction",
                patch.object(
                    mx, "compile", side_effect=RuntimeError("unsupported compiler")
                ),
            ),
        ):
            with (
                self.subTest(boundary=boundary),
                get_context().override_server_args(device="mps"),
            ):
                runner, batch, pools = self._runner_fixture()
                before = tuple(pool.clone() for pool in pools)
                with failure:
                    self.assertFalse(runner.can_run_graph(batch))
                self.assertIn(1, runner._unsupported_batches)
                self.assertFalse(runner.can_run_graph(batch))
                self.assertEqual(runner.execution_count, 0)
                for actual, expected in zip(pools, before):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_unexpected_preparation_errors_are_not_cached_as_unsupported(self):
        """Unexpected errors must surface instead of disabling a batch shape."""
        from sglang.srt.runtime_context import get_context

        for error in (
            AssertionError("model invariant"),
            AttributeError("model bug"),
            MemoryError("MPS allocation failed"),
        ):
            with (
                self.subTest(error=type(error).__name__),
                get_context().override_server_args(device="mps"),
            ):
                runner, batch, _ = self._runner_fixture()
                with patch("torch.export.export", side_effect=error):
                    with self.assertRaises(type(error)) as raised:
                        runner.can_run_graph(batch)
                self.assertIs(raised.exception, error)
                self.assertFalse(runner._unsupported_batches)
                self.assertEqual(runner.execution_count, 0)

    def test_admission_traces_without_submitting_or_evaluating_dummy_inference(self):
        import mlx.core as mx

        from sglang.srt.runtime_context import get_context

        with get_context().override_server_args(device="mps"):
            runner, batch, pools = self._runner_fixture()
            with (
                patch.object(
                    mx, "async_eval", side_effect=AssertionError("dummy submission")
                ),
                patch.object(
                    mx, "eval", side_effect=AssertionError("dummy evaluation")
                ),
            ):
                self.assertTrue(runner.can_run_graph(batch))
            self.assertEqual(runner.execution_count, 0)
            actual = runner.execute(batch)
            self.assertEqual(runner.execution_count, 1)
            self.assertTrue(torch.isfinite(actual.next_token_logits).all().item())
            expected_key = (
                runner.model_runner.model.embed(batch.input_ids)
                + batch.positions[:, None]
            )
            torch.testing.assert_close(pools[0][1], expected_key)

    def test_new_dynamic_prefill_shape_is_traced_before_admission(self):
        """A warmed prefill graph does not prove a new length can compile."""
        import mlx.core as mx

        from sglang.srt.runtime_context import get_context

        with get_context().override_server_args(device="mps"):
            runner, batch, pools = self._runner_fixture(prefill=True, tokens=3)
            self.assertTrue(runner.can_run_graph(batch))
            before = tuple(pool.clone() for pool in pools)
            batch.input_ids = torch.ones(5, device="mps", dtype=torch.int64)
            batch.positions = torch.arange(5, device="mps")
            batch.out_cache_loc = torch.arange(1, 6, device="mps")
            batch.seq_lens.fill_(5)
            batch.seq_lens_sum = batch.extend_num_tokens = 5
            batch.extend_seq_lens.fill_(5)
            batch.extend_seq_lens_cpu = [5]

            def fail_trace(*args, **kwargs):
                def trace(*flat):
                    raise RuntimeError("unsupported new length")

                return trace

            with patch.object(mx, "compile", side_effect=fail_trace):
                self.assertFalse(runner.can_run_graph(batch))
            self.assertIn(5, runner._unsupported_batches)
            self.assertNotIn(3, runner._unsupported_batches)
            for actual, expected in zip(pools, before):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_hybrid_prefill_reuses_one_dynamic_token_graph(self):
        """Prompt lengths must share the hybrid graph rather than specialize tokens."""
        import mlx.core as mx

        from sglang.kernels.ops.attention.mlx.gated_delta import (
            gdn_prefill as mlx_gdn_prefill,
        )
        from sglang.kernels.ops.attention.mlx.gated_delta_export import gdn_prefill
        from sglang.srt.hardware_backend.mps.compiled_graph import CompiledMlxGraph
        from sglang.srt.hardware_backend.mps.compiled_region import (
            discover_decode_region,
        )
        from sglang.srt.hardware_backend.mps.compiled_runner import (
            _causal_sdpa,
            _DecodeModule,
        )
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )

        torch.manual_seed(3)
        model = _ToyHybrid().eval()
        region = discover_decode_region(model)
        pools = TestHybridGdnRegion._pools(self, layers=2)
        pools["k"] = torch.randn(64, 1, HEAD_K_DIM, device="mps")
        pools["v"] = torch.randn(64, 1, HEAD_K_DIM, device="mps")
        pools["table"][1, :5] = torch.arange(1, 6, dtype=torch.int32)
        wrapper = _DecodeModule(
            model=model,
            req_pool=SimpleNamespace(req_to_token=pools["table"]),
            kv_pool=SimpleNamespace(
                get_kv_buffer=lambda _: (pools["k"], pools["v"]), page_size=1
            ),
            region=region,
            state=TestHybridGdnRegion._state(pools),
            prefill=True,
        ).eval()
        inputs = (
            torch.tensor([7, 11, 3, 5, 9], device="mps"),
            torch.arange(5, device="mps"),
            torch.tensor([1], device="mps"),
            torch.tensor([5], device="mps"),
            torch.arange(1, 6, device="mps"),
        )
        with forward_context(ForwardContext(attn_backend=wrapper.attention)):
            graph = CompiledMlxGraph(
                model=wrapper,
                example_inputs=inputs,
                custom_lowerings={
                    torch.ops.aten.scaled_dot_product_attention.default: _causal_sdpa,
                    gdn_prefill._opoverload: mlx_gdn_prefill,
                },
                dynamic_tokens=True,
            )
        self.addCleanup(graph.close)
        self.assertTrue(graph.dynamic_tokens)
        self.assertEqual(len(graph.compiled), 0)
        for count in (5, 2):
            current = (
                inputs[0][:count],
                inputs[1][:count],
                inputs[2],
                torch.tensor([count], device="mps"),
                inputs[4][:count],
            )
            outputs = graph.launch(graph.bind(current))
            mx.eval(*outputs)
            self.assertEqual(outputs[0].shape[-1], model.config.vocab_size)
        self.assertEqual(len(graph.compiled), 2)

    def test_non_mrope_model_does_not_receive_fabricated_positions(self):
        """Non-MRoPE models must see the optional position field as None."""
        from sglang.srt.hardware_backend.mps.compiled_runner import _DecodeModule
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(4, 4, device="mps")

            def forward(self, input_ids, positions, forward_batch):
                if forward_batch.mrope_positions is not None:
                    raise AssertionError("unexpected MRoPE positions")
                return LogitsProcessorOutput(
                    next_token_logits=self.embedding(input_ids)
                )

        region = SimpleNamespace(
            layers=(), linear_layers=(), position_arg="positions", pass_pp_proxy=False
        )
        wrapper = _DecodeModule(
            model=Model(),
            req_pool=SimpleNamespace(
                req_to_token=torch.zeros(1, 4, device="mps", dtype=torch.int32)
            ),
            kv_pool=SimpleNamespace(),
            region=region,
        )
        inputs = tuple(torch.tensor([1], device="mps") for _ in range(5))
        output = wrapper(*inputs)
        self.assertEqual(output[0].shape, (1, 4))

    def test_preparation_is_read_only_and_execution_errors_are_not_retried(self):
        """Successful preparation cannot commit KV or hide a later execution error."""
        from sglang.srt.runtime_context import get_context

        with get_context().override_server_args(device="mps"):
            runner, batch, pools = self._runner_fixture()
            before = tuple(pool.clone() for pool in pools)
            self.assertTrue(runner.can_run_graph(batch))
            for actual, expected in zip(pools, before):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            graph = runner._graph(batch)
            with patch.object(
                graph, "launch", side_effect=RuntimeError("execution failed")
            ):
                with self.assertRaisesRegex(RuntimeError, "execution failed"):
                    runner.execute(batch)
            self.assertNotIn(1, runner._unsupported_batches)
            self.assertEqual(runner.execution_count, 0)

    def test_dynamic_views_refresh_after_resizing_the_same_torch_objects(self):
        """Reusing storage must not reuse a stale token-dimension view."""
        from sglang.srt.hardware_backend.mps.compiled_graph import CompiledMlxGraph
        from sglang.srt.utils.tensor_bridge import mlx_to_torch

        class Model(torch.nn.Module):
            def forward(self, ids, positions, requests, lengths, locations):
                return ids + positions + locations

        storage = torch.arange(16, device="mps")
        inputs = (
            storage[:3],
            storage[:3],
            torch.tensor([0], device="mps"),
            torch.tensor([3], device="mps"),
            storage[:3],
        )
        graph = CompiledMlxGraph(
            model=Model(), example_inputs=inputs, dynamic_tokens=True
        )
        self.addCleanup(graph.close)
        for tokens in (3, 5, 2, 4):
            for index in (0, 1, 4):
                inputs[index].resize_(tokens)
            inputs[3].fill_(tokens)
            torch.mps.synchronize()
            actual = mlx_to_torch(graph.launch(graph.bind(inputs))[0])
            torch.testing.assert_close(actual, Model()(*inputs), rtol=0, atol=0)

    def test_cpu_fp64_binding_does_not_silently_downcast(self):
        """The graph's CPU branch must preserve values below fp32 resolution."""
        import mlx.core as mx

        from sglang.srt.hardware_backend.mps.compiled_graph import CompiledMlxGraph
        from sglang.srt.utils.tensor_bridge import mlx_to_torch

        class Model(torch.nn.Module):
            def forward(self, x):
                return x * 2

        inputs = (torch.tensor([1 + 2**-40], dtype=torch.float64),)
        graph = CompiledMlxGraph(model=Model(), example_inputs=inputs)
        self.addCleanup(graph.close)
        with mx.stream(mx.cpu):
            output = graph.launch(graph.bind(inputs))[0]
            actual = mlx_to_torch(output, device="cpu")
        torch.testing.assert_close(actual, Model()(*inputs), rtol=0, atol=0)

    def test_dynamic_shape_specialization_is_an_eager_fallback(self):
        """Length-specific branches must not abort dynamic-prefill initialization."""
        from sglang.srt.hardware_backend.mps.compiled_graph import CompiledMlxGraph
        from sglang.srt.hardware_backend.mps.compiled_region import UnsupportedMlxRegion

        class Model(torch.nn.Module):
            def forward(self, ids, positions, requests, lengths, locations):
                return ids + (1 if ids.numel() > 8 else 0)

        inputs = (
            torch.arange(8, device="mps"),
            torch.arange(8, device="mps"),
            torch.tensor([0], device="mps"),
            torch.tensor([8], device="mps"),
            torch.arange(8, device="mps"),
        )
        with self.assertRaisesRegex(UnsupportedMlxRegion, "dynamic MLX export"):
            CompiledMlxGraph(model=Model(), example_inputs=inputs, dynamic_tokens=True)

    def test_prefill_export_preserves_last_logits_and_private_kv(self):
        from sglang.srt.hardware_backend.mps.compiled_graph import CompiledMlxGraph
        from sglang.srt.hardware_backend.mps.compiled_region import (
            UnsupportedMlxRegion,
            discover_decode_region,
        )
        from sglang.srt.hardware_backend.mps.compiled_runner import (
            _causal_sdpa,
            _DecodeModule,
        )
        from sglang.srt.layers.logits_processor import LogitsProcessor
        from sglang.srt.layers.radix_attention import RadixAttention
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )
        from sglang.srt.runtime_context import get_context
        from sglang.srt.utils.tensor_bridge import mlx_to_torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(vocab_size=17)
                self.embed = torch.nn.Embedding(17, 64, device="mps")
                self.layers = torch.nn.ModuleList([RadixAttention(1, 64, 0.125, 1, 0)])
                self.lm_head = torch.nn.Linear(64, 17, device="mps")
                self.processor = LogitsProcessor(self.config)
                self.mutate = False

            def forward(self, input_ids, positions, forward_batch):
                if self.mutate:
                    forward_batch.extend_prefix_lens_cpu[0] = 1
                hidden = self.embed(input_ids) + positions[:, None]
                hidden = self.layers[0](hidden, hidden, hidden, forward_batch)
                return self.processor(input_ids, hidden, self.lm_head, forward_batch)

        with get_context().override_server_args(device="mps", enable_fp32_lm_head=True):
            model = Model().eval()
            wrapper = _DecodeModule(
                model=model,
                req_pool=SimpleNamespace(),
                kv_pool=SimpleNamespace(page_size=1),
                region=discover_decode_region(model),
                prefill=True,
            )
            inputs = (
                torch.tensor([1, 2, 3], device="mps"),
                torch.arange(3, device="mps"),
                torch.tensor([1], device="mps"),
                torch.tensor([3], device="mps"),
                torch.tensor([5, 6, 7], device="mps"),
            )
            lowerings = {
                torch.ops.aten.scaled_dot_product_attention.default: _causal_sdpa
            }
            with forward_context(ForwardContext(attn_backend=wrapper.attention)):
                graph = CompiledMlxGraph(
                    model=wrapper,
                    example_inputs=inputs,
                    custom_lowerings=lowerings,
                    dynamic_tokens=True,
                )
                self.addCleanup(graph.close)
                for ids in (
                    [1, 2, 3],
                    [4, 5],
                    [1, 2, 3, 4, 5],
                    [1, 2, 3, 4],
                    [1, 2, 3, 4, 5, 6],
                    [4, 5, 6],
                ):
                    current = (
                        torch.tensor(ids, device="mps"),
                        torch.arange(len(ids), device="mps"),
                        inputs[2],
                        torch.tensor([len(ids)], device="mps"),
                        torch.arange(5, 5 + len(ids), device="mps"),
                    )
                    wrapper.static_reads.clear()
                    expected = wrapper(*current)
                    torch.mps.synchronize()
                    actual = graph.launch(graph.bind(current))
                    for lhs, rhs in zip(actual, expected):
                        torch.testing.assert_close(
                            mlx_to_torch(lhs), rhs, rtol=0.02, atol=0.02
                        )
                    self.assertLessEqual(len(graph.compiled), 4)
                for candidate in (
                    (inputs[0], inputs[1][:-1], *inputs[2:]),
                    (inputs[0].float(), *inputs[1:]),
                    (torch.arange(6, device="mps")[::2], *inputs[1:]),
                ):
                    with self.assertRaises(ValueError):
                        graph.bind(candidate)
                model.mutate = True
                with self.assertRaisesRegex(
                    UnsupportedMlxRegion, "mutate batch metadata"
                ):
                    CompiledMlxGraph(
                        model=wrapper, example_inputs=inputs, custom_lowerings=lowerings
                    )

    def test_prefill_long_admission_and_authoritative_kv_mapping(self):
        from sglang.srt.hardware_backend.mps.compiled_runner import (
            CompiledMlxRunner,
            _plain_prefill,
        )
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.model_executor.forward_mode import ForwardMode

        capacity = 33024
        table = torch.zeros(2, capacity, device="mps", dtype=torch.int32)
        keys = torch.zeros(capacity + 1, 1, 64, device="mps")
        runner = object.__new__(CompiledMlxRunner)
        runner._prefill = True
        runner._layers = (SimpleNamespace(layer_id=0),)
        runner._linear_layers = ()
        runner._state = None
        runner.model_runner = SimpleNamespace(
            model=SimpleNamespace(config=SimpleNamespace(vocab_size=17)),
            model_config=SimpleNamespace(context_len=capacity),
            req_to_token_pool=SimpleNamespace(req_to_token=table),
            token_to_kv_pool=SimpleNamespace(
                page_size=1, get_key_buffer=lambda _: keys
            ),
        )
        for tokens in (16384, 32768):
            with self.subTest(tokens=tokens):
                locations = torch.arange(1, tokens + 1, device="mps")
                table[1, :tokens] = locations.to(torch.int32)
                batch = ForwardBatch(
                    forward_mode=ForwardMode.EXTEND,
                    batch_size=1,
                    input_ids=torch.ones(tokens, device="mps", dtype=torch.int64),
                    positions=torch.arange(tokens, device="mps"),
                    req_pool_indices=torch.tensor([1], device="mps"),
                    seq_lens=torch.tensor([tokens], device="mps"),
                    seq_lens_sum=tokens,
                    out_cache_loc=locations,
                    extend_num_tokens=tokens,
                    extend_seq_lens=torch.tensor([tokens], device="mps"),
                    extend_seq_lens_cpu=[tokens],
                    extend_prefix_lens=torch.tensor([0], device="mps"),
                    extend_prefix_lens_cpu=[0],
                    extend_start_loc=torch.tensor([0], device="mps"),
                )
                self.assertTrue(_plain_prefill(batch))
                runner._validate_prefill_metadata(batch)
                table[1, tokens - 1] = 1
                with self.assertRaisesRegex(ValueError, "KV mapping"):
                    runner._validate_prefill_metadata(batch)
                batch.extend_prefix_lens_cpu = [1]
                self.assertFalse(_plain_prefill(batch))
                batch.extend_prefix_lens_cpu = [0]
                batch.batch_size = 2
                self.assertFalse(_plain_prefill(batch))
                batch.batch_size = 1
                batch.mm_inputs = [SimpleNamespace()]
                self.assertFalse(_plain_prefill(batch))

    def test_radix_lowering_preserves_page_size_and_rejects_broken_pages(self):
        from sglang.kernels.ops.attention.mlx.radix_attention import radix_decode
        from sglang.kernels.ops.attention.mlx.radix_attention_export import (
            radix_decode as reference,
        )
        from sglang.kernels.ops.attention.mlx.radix_attention_export import (
            validate_page_slots,
        )
        from sglang.srt.hardware_backend.mps.compiled_graph import CompiledMlxGraph
        from sglang.srt.utils.tensor_bridge import mlx_to_torch

        class Attention(torch.nn.Module):
            def forward(self, q, k, v, pool, table, rows, lengths):
                return reference(
                    q, k, v, pool, pool, table, rows, lengths, 0.125, page_size=16
                )

        def lower(*args, page_size, tails):
            return radix_decode(*args, page_size=page_size, tails=tails)

        inputs = (
            torch.ones((1, 4, 64), device="mps"),
            torch.ones((1, 2, 64), device="mps"),
            torch.full((1, 2, 64), 3.0, device="mps"),
            torch.ones((64, 2, 64), device="mps"),
            torch.cat((torch.arange(32, 48), torch.arange(16, 32)))
            .to(device="mps", dtype=torch.int32)
            .unsqueeze(0),
            torch.tensor([0], device="mps"),
            torch.tensor([17], device="mps"),
        )
        model = Attention()
        expected = model(*inputs)
        graph = CompiledMlxGraph(model=model, example_inputs=inputs, attention=lower)
        self.addCleanup(graph.close)
        actual = mlx_to_torch(graph.launch(graph.bind(inputs))[0])
        torch.testing.assert_close(actual, expected)
        slots = inputs[4][0].cpu()
        for index, value in ((0, 33), (15, 16), (16, 17)):
            broken = slots.clone()
            broken[index] = value
            with (
                self.subTest(index=index),
                self.assertRaisesRegex(ValueError, "aligned, contiguous"),
            ):
                validate_page_slots(broken, 16)

    def test_rms_fusions_preserve_cast_boundaries(self):
        from sglang.srt.hardware_backend.mps.compiled_graph import CompiledMlxGraph
        from sglang.srt.utils.tensor_bridge import mlx_to_torch

        class Norm(torch.nn.Module):
            def __init__(self, weighted, cast_before_weight):
                super().__init__()
                self.weight = (
                    torch.nn.Parameter(
                        torch.randn(128, device="mps", dtype=torch.bfloat16)
                    )
                    if weighted
                    else None
                )
                self.cast_before_weight = cast_before_weight

            def forward(self, value):
                x = value.float()
                normalized = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
                if self.weight is None:
                    return normalized.to(value.dtype)
                if self.cast_before_weight:
                    normalized = normalized.to(value.dtype)
                return (normalized * self.weight).to(value.dtype)

        cases = [
            (dtype, False, False)
            for dtype in (torch.float32, torch.float16, torch.bfloat16)
        ] + [(torch.bfloat16, True, cast) for cast in (False, True)]
        for dtype, weighted, cast_before_weight in cases:
            with self.subTest(
                dtype=dtype, weighted=weighted, cast_before_weight=cast_before_weight
            ):
                inputs = (torch.randn(3, 128, device="mps", dtype=dtype),)
                model = Norm(weighted, cast_before_weight)
                graph = CompiledMlxGraph(model=model, example_inputs=inputs)
                self.addCleanup(graph.close)
                if weighted:
                    self.assertEqual(
                        len(graph.weighted_norms), int(not cast_before_weight)
                    )
                else:
                    self.assertEqual(len(graph.norms), 1)
                torch.mps.synchronize()
                actual = mlx_to_torch(graph.launch(graph.bind(inputs))[0])
                torch.testing.assert_close(actual, model(*inputs))

    def test_whole_forward_preserves_logits_and_checks_attention_and_metadata(self):
        from sglang.kernels.ops.attention.mlx.radix_attention import radix_decode
        from sglang.srt.hardware_backend.mps.compiled_graph import CompiledMlxGraph
        from sglang.srt.hardware_backend.mps.compiled_region import (
            UnsupportedMlxRegion,
            discover_decode_region,
            static_metadata_matches,
        )
        from sglang.srt.hardware_backend.mps.compiled_runner import _DecodeModule
        from sglang.srt.layers.logits_processor import LogitsProcessor
        from sglang.srt.layers.radix_attention import RadixAttention
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )
        from sglang.srt.model_executor.forward_mode import ForwardMode
        from sglang.srt.runtime_context import get_context
        from sglang.srt.utils.tensor_bridge import mlx_to_torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(vocab_size=17)
                self.embed = torch.nn.Embedding(
                    17, 64, device="mps", dtype=torch.bfloat16
                )
                self.layers = torch.nn.ModuleList(
                    [RadixAttention(1, 64, 0.125, 1, i) for i in range(2)]
                )
                self.lm_head = torch.nn.Linear(
                    64, 17, device="mps", dtype=torch.bfloat16
                )
                self.logits_processor = LogitsProcessor(self.config, logit_scale=1.75)
                self.repeat_attention = False
                self.read_sum = False
                self.mutate_metadata = False

            def forward(self, input_ids, forward_batch, positions):
                if self.mutate_metadata:
                    forward_batch.return_logprob = True
                hidden = self.embed(input_ids) + positions[:, None].to(torch.bfloat16)
                if self.read_sum:
                    hidden = hidden + forward_batch.seq_lens_sum
                for layer in self.layers:
                    hidden = layer(hidden, hidden, hidden, forward_batch)
                if self.repeat_attention:
                    hidden = self.layers[-1](hidden, hidden, hidden, forward_batch)
                return self.logits_processor(
                    input_ids, hidden, self.lm_head, forward_batch
                )

        with get_context().override_server_args(
            device="mps", tp_size=1, pp_size=1, enable_fp32_lm_head=True
        ):
            model = Model().eval()
            pools = tuple(
                torch.zeros(8, 1, 64, device="mps", dtype=torch.bfloat16)
                for _ in range(4)
            )
            table = torch.tensor([[1, 2, 3, 4]], device="mps", dtype=torch.int32)
            wrapper = _DecodeModule(
                model=model,
                req_pool=SimpleNamespace(req_to_token=table),
                kv_pool=SimpleNamespace(
                    get_kv_buffer=lambda i: pools[2 * i : 2 * i + 2],
                    page_size=1,
                ),
                region=discover_decode_region(model),
            )
            inputs = tuple(
                torch.tensor([value], device="mps", dtype=torch.int64)
                for value in (3, 0, 0, 1, 1)
            )
            with forward_context(ForwardContext(attn_backend=wrapper.attention)):
                graph = CompiledMlxGraph(
                    model=wrapper, example_inputs=inputs, attention=radix_decode
                )
                self.addCleanup(graph.close)
                for token, position in ((3, 0), (4, 1)):
                    current = (
                        torch.full_like(inputs[0], token),
                        torch.full_like(inputs[1], position),
                        *inputs[2:],
                    )
                    expected = wrapper(*current)[0]
                    torch.mps.synchronize()
                    actual = mlx_to_torch(graph.launch(graph.bind(current))[0])
                    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
                model.repeat_attention = True
                with self.assertRaisesRegex(UnsupportedMlxRegion, "invocation order"):
                    CompiledMlxGraph(
                        model=wrapper, example_inputs=inputs, attention=radix_decode
                    )
                model.repeat_attention = False
                model.read_sum = True
                wrapper(*inputs)
                model.mutate_metadata = True
                with self.assertRaisesRegex(
                    UnsupportedMlxRegion, "mutate batch metadata"
                ):
                    CompiledMlxGraph(
                        model=wrapper, example_inputs=inputs, attention=radix_decode
                    )
            batch = ForwardBatch(
                forward_mode=ForwardMode.DECODE,
                batch_size=1,
                input_ids=inputs[0],
                req_pool_indices=inputs[2],
                seq_lens=inputs[3],
                out_cache_loc=inputs[4],
                seq_lens_sum=1,
            )
            self.assertIn("seq_lens_sum", wrapper.static_reads)
            self.assertFalse(static_metadata_matches(batch, wrapper.static_reads))

    def test_silu_does_not_round_sigmoid_before_multiplication(self):
        from sglang.srt.hardware_backend.mps.compiled_graph import CompiledMlxGraph
        from sglang.srt.utils.tensor_bridge import mlx_to_torch

        inputs = (torch.linspace(-8, 8, 4096, device="mps", dtype=torch.bfloat16),)
        model = torch.nn.SiLU()
        graph = CompiledMlxGraph(model=model, example_inputs=inputs)
        self.addCleanup(graph.close)
        torch.mps.synchronize()
        actual = mlx_to_torch(graph.launch(graph.bind(inputs))[0])
        torch.testing.assert_close(actual, model(*inputs), rtol=0, atol=0)

    def test_structural_admission_rejects_ambiguous_or_reused_state(self):
        from sglang.srt.hardware_backend.mps.compiled_region import (
            UnsupportedMlxRegion,
            discover_decode_region,
        )
        from sglang.srt.layers.radix_attention import RadixAttention

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = torch.nn.ModuleDict(
                    {
                        "blocks": torch.nn.ModuleList(
                            [
                                RadixAttention(
                                    num_heads=2,
                                    head_dim=64,
                                    scaling=0.125,
                                    num_kv_heads=1,
                                    layer_id=i,
                                )
                                for i in range(2)
                            ]
                        )
                    }
                )

            def forward(self, input_ids, forward_batch, position_ids):
                return input_ids

        model = Model()
        region = discover_decode_region(model)
        self.assertEqual(region.position_arg, "position_ids")
        self.assertEqual(tuple(x.layer_id for x in region.layers), (0, 1))
        from sglang.srt.hardware_backend.mps.compiled_graph import CompiledMlxGraph
        from sglang.srt.hardware_backend.mps.compiled_runner import CompiledMlxRunner

        runner = object.__new__(CompiledMlxRunner)
        runner.model_runner = SimpleNamespace(model=model)
        runner._model_structure = None
        runner._graphs = {}
        runner._unsupported_batches = {}
        runner._pending = None
        runner._refresh_region()
        graph = CompiledMlxGraph(
            model=torch.nn.Linear(4, 4, device="mps"),
            example_inputs=(torch.ones(1, 4, device="mps"),),
        )
        self.addCleanup(graph.close)
        runner._graphs[1] = graph
        added = torch.nn.Linear(4, 4, device="mps")
        model.transformer["blocks"][0].extra = added
        runner._refresh_region()
        self.assertTrue(graph.closed)
        self.assertFalse(runner._graphs)
        self.assertIn((added, "weight", True), runner._weight_sources)
        model.transformer["blocks"][1].layer_id = 0
        with self.assertRaisesRegex(UnsupportedMlxRegion, "unique"):
            discover_decode_region(model)
        model = Model()
        model.other = Model().transformer
        with self.assertRaisesRegex(UnsupportedMlxRegion, "one decoder stack"):
            discover_decode_region(model)
        model = Model()
        model.transformer["blocks"][1] = model.transformer["blocks"][0]
        with self.assertRaisesRegex(UnsupportedMlxRegion, "unique"):
            discover_decode_region(model)

        class UnknownArgument(Model):
            def forward(self, input_ids, positions, forward_batch, secret_state):
                return input_ids

        with self.assertRaisesRegex(UnsupportedMlxRegion, "secret_state"):
            discover_decode_region(UnknownArgument())

    def test_dense_activation_and_layernorm_lowerings_match_torch(self):
        from sglang.srt.hardware_backend.mps.compiled_graph import CompiledMlxGraph
        from sglang.srt.utils.tensor_bridge import mlx_to_torch

        class Model(torch.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.norm = torch.nn.LayerNorm(64, device="mps", dtype=dtype)

            def forward(self, x):
                original = x
                x = self.norm(x)
                return (
                    torch.nn.functional.gelu(x),
                    torch.nn.functional.gelu(x, approximate="tanh"),
                    torch.relu(x),
                    torch.tanh(x),
                    *torch.native_layer_norm(
                        original, (64,), self.norm.weight, self.norm.bias, self.norm.eps
                    ),
                )

        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = Model(dtype).eval()
                inputs = (torch.randn(3, 64, device="mps", dtype=dtype),)
                graph = CompiledMlxGraph(model=model, example_inputs=inputs)
                self.addCleanup(graph.close)
                torch.mps.synchronize()
                outputs = graph.launch(graph.bind(inputs))
                for actual, expected in zip(outputs, model(*inputs)):
                    torch.testing.assert_close(mlx_to_torch(actual), expected)

    def test_shared_sdpa_lowering_extends_compiled_regions(self):
        from sglang.srt.hardware_backend.mps.compiled_graph import CompiledMlxGraph
        from sglang.srt.utils.tensor_bridge import mlx_to_torch

        class Model(torch.nn.Module):
            def forward(self, q, k, v):
                return torch.nn.functional.scaled_dot_product_attention(q, k, v)

        model = Model().eval()
        inputs = tuple(torch.randn(2, 4, 8, 32, device="mps") for _ in range(3))
        graph = CompiledMlxGraph(model=model, example_inputs=inputs)
        self.addCleanup(graph.close)
        torch.mps.synchronize()
        actual = mlx_to_torch(graph.launch(graph.bind(inputs))[0])
        torch.testing.assert_close(actual, model(*inputs))

    def test_custom_metal_lowering_survives_export_and_dependent_compile(self):
        import mlx.core as mx

        from sglang.srt.hardware_backend.mps.compiled_graph import CompiledMlxGraph
        from sglang.srt.utils.tensor_bridge import mlx_to_torch

        @torch.library.custom_op(
            "sglang_test_compiled::square_plus_one", mutates_args=()
        )
        def custom(x: torch.Tensor) -> torch.Tensor:
            return x.square() + 1

        @custom.register_fake
        def fake(x):
            return torch.empty_like(x)

        class Model(torch.nn.Module):
            def forward(self, x):
                return custom(x) * 0.5

        kernel = mx.fast.metal_kernel(
            name="test_compiled_square_plus_one",
            input_names=["x"],
            output_names=["out"],
            source="uint i = thread_position_in_grid.x; out[i] = x[i] * x[i] + 1;",
        )

        def lower(x):
            return kernel(
                inputs=[x],
                grid=(x.size, 1, 1),
                threadgroup=(32, 1, 1),
                output_shapes=[x.shape],
                output_dtypes=[x.dtype],
            )[0]

        model = Model().eval()
        inputs = (torch.randn(2, 32, device="mps"),)
        with self.assertRaisesRegex(ValueError, "no MLX lowering"):
            CompiledMlxGraph(model=model, example_inputs=inputs)
        torch.mps.synchronize()
        graph = CompiledMlxGraph(
            model=model,
            example_inputs=inputs,
            custom_lowerings={
                torch.ops.sglang_test_compiled.square_plus_one.default: lower
            },
        )
        self.addCleanup(graph.close)
        arrays = graph.bind(inputs)
        first = graph.launch(arrays)
        second = graph.launch(first)
        torch.testing.assert_close(mlx_to_torch(second[0]), model(model(*inputs)))

    def test_host_alias_retains_offset_storage_and_live_updates(self):
        from sglang.srt.hardware_backend.mps.compiled_graph import host_alias

        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                source = torch.arange(24, dtype=dtype, device="mps").reshape(6, 4)
                view = source[2:5]
                torch.mps.synchronize()
                alias = host_alias(view)
                source.add_(7)
                torch.mps.synchronize()
                torch.testing.assert_close(alias, view.cpu())
                expected = alias.clone()
                del source, view
                gc.collect()
                torch.testing.assert_close(alias, expected)

    def test_async_requires_compiled_local_mps_execution(self):
        from sglang.srt.server_args import ServerArgs

        self.assertEqual(ServerArgs(model_path="dummy").mps_execution_backend, "eager")
        for option in (
            {"enable_hierarchical_cache": True},
            {"enable_lmcache": True},
            {"enable_session_radix_cache": True},
            {"disable_overlap_schedule": False},
        ):
            with (
                self.subTest(option=option),
                self.assertRaisesRegex(ValueError, "local"),
            ):
                ServerArgs(
                    model_path="dummy",
                    device="mps",
                    mps_execution_backend="mlx-compiled",
                    **{"disable_overlap_schedule": True, **option},
                ).resolve_once()
        with self.assertRaisesRegex(ValueError, "speculative"):
            ServerArgs(
                model_path="dummy",
                device="mps",
                mps_execution_backend="mlx-compiled",
                disable_overlap_schedule=True,
                speculative_algorithm="EAGLE",
            ).resolve_once()
        with self.assertRaisesRegex(ValueError, "standard Torch MPS"):
            ServerArgs(
                model_path="dummy",
                device="cpu",
                mps_execution_backend="mlx-compiled",
            ).resolve_once()

    def test_mutation_and_changed_attribute_metadata_fail_closed(self):
        from sglang.srt.hardware_backend.mps.compiled_graph import CompiledMlxGraph

        class Mutating(torch.nn.Module):
            def forward(self, x):
                return x.add_(1)

        with self.assertRaisesRegex(ValueError, "mutate"):
            CompiledMlxGraph(
                model=Mutating(), example_inputs=(torch.ones(4, device="mps"),)
            )
        model = torch.nn.Linear(4, 4, device="mps").eval()
        inputs = (torch.ones(1, 4, device="mps"),)
        graph = CompiledMlxGraph(model=model, example_inputs=inputs)
        graph.bind(inputs)
        weight = model.weight
        for replacement in (
            weight.T,
            weight.to(torch.float16),
            weight.cpu(),
            torch.ones(8, 4, device="mps"),
        ):
            model.weight = torch.nn.Parameter(replacement)
            with self.subTest(signature=(replacement.shape, replacement.stride())):
                with self.assertRaisesRegex(ValueError, "attribute metadata"):
                    graph.bind(inputs)
        model.weight = weight
        with self.assertRaisesRegex(ValueError, "input shape, dtype, stride"):
            graph.bind((torch.ones(1, 8, device="mps")[:, ::2],))
        graph.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            graph.bind(inputs)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            graph.launch(())

    def test_enqueue_precedes_current_wait_and_stale_results_are_discarded(self):
        import mlx.core as mx

        from sglang.srt.hardware_backend.mps.compiled_graph import CompiledMlxGraph
        from sglang.srt.hardware_backend.mps.compiled_runner import (
            CompiledMlxRunner,
            _batch_inputs,
        )
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.model_executor.forward_mode import ForwardMode

        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(4, 4, device="mps")

            def forward(self, ids, positions, requests, lengths, locations):
                logits = self.embedding(ids)
                kv = logits.reshape(1, -1, 1, 4)
                return logits, kv, kv

        model = Decoder().eval()
        table = torch.zeros(4, 32, device="mps", dtype=torch.int32)
        table[1, 1] = 5
        batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=1,
            input_ids=torch.tensor([1], device="mps"),
            positions=torch.tensor([1], device="mps"),
            req_pool_indices=torch.tensor([1], device="mps"),
            seq_lens=torch.tensor([2], device="mps"),
            seq_lens_sum=2,
            out_cache_loc=torch.tensor([5], device="mps"),
        )
        batch.sampling_info = SimpleNamespace(
            is_all_greedy=True,
            grammars=None,
            grammar_mask=None,
            has_custom_logit_processor=False,
            logit_bias=None,
            acc_additive_penalties=None,
            acc_scaling_penalties=None,
            penalizer_orchestrator=None,
        )
        graph = CompiledMlxGraph(model=model, example_inputs=_batch_inputs(batch))
        self.addCleanup(graph.close)
        runner = object.__new__(CompiledMlxRunner)
        runner._prefill = False
        runner.model_runner = SimpleNamespace(
            model=model,
            model_config=SimpleNamespace(context_len=32),
            req_to_token_pool=SimpleNamespace(
                req_to_token=table, req_generation=torch.zeros(4, dtype=torch.int64)
            ),
        )
        runner._max_batch_size = 4
        runner._unsupported_reason = None
        runner._unsupported_batches = {}
        runner._model_structure = (
            model,
            tuple(
                (module, tuple(module._modules.items())) for module in model.modules()
            ),
        )
        runner._weight_sources = ((model.embedding, "weight", True),)
        model.static_reads = {}
        runner._fallback_reasons = set()
        runner._layers = ()
        runner._linear_layers = ()
        runner._state = None
        runner._async = False
        runner._pending = None
        runner.execution_count = runner.prefetch_count = 0
        runner.prefetch_hits = runner.prefetch_discards = 0
        runner.prefetch_discard_reasons = {}
        events = []
        launch, evaluate = graph.launch, mx.eval

        torch.mps.synchronize()
        synchronous = graph.launch(graph.bind(_batch_inputs(batch)))
        runner._prefetch(
            graph=graph,
            arrays=graph.bind(_batch_inputs(batch)),
            outputs=synchronous,
            batch=batch,
            metadata=runner._metadata(batch),
            weights=runner._weights(),
        )
        self.assertIsNone(runner._pending)
        from sglang.srt.runtime_context import get_context

        with get_context().override_server_args(disable_overlap_schedule=False):
            with self.assertRaisesRegex(RuntimeError, "normal scheduler"):
                runner.enable_lookahead()
            self.assertFalse(runner._async)
        with get_context().override_server_args(disable_overlap_schedule=True):
            runner.enable_lookahead()

        def record_launch(*args, **kwargs):
            events.append("launch")
            return launch(*args, **kwargs)

        def record_wait(*args, **kwargs):
            events.append("wait")
            return evaluate(*args, **kwargs)

        with (
            patch.object(runner, "_graph", return_value=graph),
            patch.object(
                runner,
                "_validate_metadata",
                side_effect=lambda _: torch.mps.synchronize(),
            ),
            patch.object(graph, "launch", side_effect=record_launch),
            patch.object(mx, "eval", side_effect=record_wait),
        ):
            result = runner.execute(batch)
        self.assertEqual(events[:3], ["launch", "launch", "wait"])
        torch.testing.assert_close(
            result.next_token_logits, model.embedding(batch.input_ids)
        )
        runner.drain()
        pending = runner._pending
        runner.model_runner.model_config.context_len = 2
        runner._pending = None
        count = runner.prefetch_count
        runner._prefetch(
            graph=graph,
            arrays=graph.bind(_batch_inputs(batch)),
            outputs=pending.outputs,
            batch=batch,
            metadata=runner._metadata(batch),
            weights=runner._weights(),
        )
        self.assertIsNone(runner._pending)
        self.assertEqual(runner.prefetch_count, count)
        runner.model_runner.model_config.context_len = 32
        runner._pending = pending
        batch.input_ids = torch.tensor(pending.tokens.tolist(), device="mps")
        batch.positions = batch.positions + 1
        batch.seq_lens = batch.seq_lens + 1
        torch.mps.synchronize()
        metadata, weights = runner._metadata(batch), runner._weights()
        self.assertIs(
            runner._consume(
                graph=graph, batch=batch, metadata=metadata, weights=weights
            ),
            pending.outputs,
        )
        self.assertEqual(runner.prefetch_hits, 1)
        for changed in ("generation", "length", "weights", "tokens", "prefix_slot"):
            with self.subTest(changed=changed):
                runner._pending = pending
                candidate, stamp = metadata, weights
                if changed == "generation":
                    candidate = (*metadata[:3], (1,))
                elif changed == "length":
                    candidate = (metadata[0], (4,), *metadata[2:])
                elif changed == "weights":
                    original = model.embedding.weight.detach().clone()
                    with torch.no_grad():
                        model.embedding.weight.add_(1)
                    stamp = runner._weights()
                    with torch.no_grad():
                        model.embedding.weight.copy_(original)
                elif changed == "tokens":
                    batch.input_ids = (batch.input_ids + 1) % 4
                else:
                    table[1, 1] = 6
                torch.mps.synchronize()
                self.assertIsNone(
                    runner._consume(
                        graph=graph, batch=batch, metadata=candidate, weights=stamp
                    )
                )
                batch.input_ids = torch.tensor(pending.tokens.tolist(), device="mps")
        self.assertEqual(runner.prefetch_discards, 5)
        self.assertEqual(
            runner.prefetch_discard_reasons,
            {"metadata": 2, "weights": 1, "tokens": 1, "cache_mapping": 1},
        )
        runner._pending = pending
        batch.forward_mode = ForwardMode.EXTEND
        self.assertFalse(runner.can_run_graph(batch))
        self.assertIsNone(runner._pending)

    def test_embedding_unbounded_slice_and_live_nonpersistent_buffer(self):
        from sglang.srt.hardware_backend.mps.compiled_graph import CompiledMlxGraph
        from sglang.srt.utils.tensor_bridge import mlx_to_torch

        class Embedding(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(16, 8, device="mps")
                self.register_buffer(
                    "offset", torch.ones(8, device="mps"), persistent=False
                )

            def forward(self, ids):
                result = torch.ops.aten.alias.default(self.embedding(ids)[:, :])
                return result + self.offset

        model = Embedding().eval()
        inputs = (torch.tensor([1, 3], device="mps"),)
        with torch.no_grad():
            graph = CompiledMlxGraph(model=model, example_inputs=inputs)
            for replacement in (False, True):
                if replacement:
                    model.offset = torch.full((8,), 3.0, device="mps")
                torch.mps.synchronize()
                actual = graph.launch(graph.bind(inputs))
                torch.testing.assert_close(mlx_to_torch(actual[0]), model(*inputs))
            graph.close()

    def test_aliased_module_bindings_follow_submodule_replacement(self):
        """Exported alias paths must survive enumeration and subtree replacement."""
        from sglang.srt.hardware_backend.mps.compiled_graph import CompiledMlxGraph
        from sglang.srt.utils.tensor_bridge import mlx_to_torch

        class Branch(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4, device="mps")
                self.register_buffer(
                    "offset", torch.randn(4, device="mps"), persistent=False
                )

            def forward(self, x):
                return self.linear(x) + self.offset

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.left = Branch()
                self.right = self.left

            def forward(self, x):
                return self.right(x)

        model = Model().eval()
        inputs = (torch.randn(2, 4, device="mps"),)
        with torch.no_grad():
            graph = CompiledMlxGraph(model=model, example_inputs=inputs)
            self.addCleanup(graph.close)
            for replace in (False, True):
                with self.subTest(replace=replace):
                    if replace:
                        model.right = Branch()
                    torch.mps.synchronize()
                    output = graph.launch(graph.bind(inputs))
                    actual, expected = mlx_to_torch(output[0]), model(*inputs)

                    def failure_details(message):
                        import platform
                        import subprocess
                        from copy import deepcopy
                        from importlib.metadata import version

                        cpu = (
                            deepcopy(model)
                            .cpu()
                            .double()(*[x.cpu().double() for x in inputs])
                        )
                        linear = model.right.linear
                        without_bias = torch.nn.functional.linear(
                            inputs[0], linear.weight, None
                        )
                        explicit_bias = without_bias + linear.bias + model.right.offset
                        components = {
                            "input": inputs[0].cpu().tolist(),
                            "weight": linear.weight.cpu().tolist(),
                            "bias": linear.bias.cpu().tolist(),
                            "offset": model.right.offset.cpu().tolist(),
                            "mps_linear": linear(inputs[0]).cpu().tolist(),
                            "mps_no_bias": without_bias.cpu().tolist(),
                            "mps_explicit_bias_and_offset": explicit_bias.cpu().tolist(),
                            "eager_minus_cpu": (expected.cpu().double() - cpu).tolist(),
                        }
                        return (
                            f"{message}\nreplace={replace}, hardware={platform.platform()}, "
                            f"chip={subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()}, "
                            f"torch={torch.__version__}, torch_git={torch.version.git_version}, "
                            f"mlx={version('mlx')}\nactual={actual.cpu().tolist()}\n"
                            f"eager={expected.cpu().tolist()}\ncpu_fp64={cpu.tolist()}\n"
                            f"components={components}\n"
                            f"bindings={[(i, str(view.torch_tensor.dtype), tuple(view.torch_tensor.shape), tuple(view.torch_tensor.stride()), view.torch_tensor.storage_offset()) for i, view in graph.views.items()]}"
                        )

                    torch.testing.assert_close(actual, expected, msg=failure_details)

    def test_compiled_live_weights_and_dependent_launch(self):
        import mlx.core as mx

        from sglang.srt.hardware_backend.mps.compiled_graph import CompiledMlxGraph
        from sglang.srt.utils.tensor_bridge import mlx_to_torch

        model = (
            torch.nn.Sequential(
                torch.nn.Linear(32, 32), torch.nn.SiLU(), torch.nn.Linear(32, 32)
            )
            .to("mps")
            .eval()
        )
        inputs = (torch.randn(2, 32, device="mps"),)
        with torch.no_grad():
            graph = CompiledMlxGraph(model=model, example_inputs=inputs)
            torch.mps.synchronize()
            flat = graph.bind(inputs)
            first = graph.launch(flat)
            following = list(flat)
            user_index = next(
                i
                for i, (kind, _) in enumerate(graph.bindings)
                if kind.name == "USER_INPUT"
            )
            following[user_index] = first[0]
            second = graph.launch(tuple(following))
            # Both dependent calls were submitted before any host materialization.
            mx.eval(*second)
            torch.testing.assert_close(
                mlx_to_torch(second[0]), model(model(*inputs)), atol=1e-5, rtol=1e-4
            )
            model[0].weight.add_(0.1)
            torch.mps.synchronize()
            updated = graph.launch(graph.bind(inputs))
            torch.testing.assert_close(
                mlx_to_torch(updated[0]), model(*inputs), atol=1e-5, rtol=1e-4
            )
            model[0].weight = torch.nn.Parameter(model[0].weight * 0.5)
            torch.mps.synchronize()
            replaced = graph.launch(graph.bind(inputs))
            exported = mlx_to_torch(replaced[0])
            expected = model(*inputs).clone()
            graph.close()
            torch.testing.assert_close(exported, expected, atol=1e-5, rtol=1e-4)


# GDN geometry with a head group of two, like the 9B checkpoint.
NUM_K_HEADS, HEAD_K_DIM, NUM_V_HEADS, HEAD_V_DIM, KERNEL = 2, 128, 4, 128, 4
CHANNELS = 2 * NUM_K_HEADS * HEAD_K_DIM + NUM_V_HEADS * HEAD_V_DIM
GEOMETRY = dict(num_k_heads=NUM_K_HEADS, head_k_dim=HEAD_K_DIM, head_v_dim=HEAD_V_DIM)


def _mlx():
    import mlx.core as mx

    return mx


def _to_torch(array):
    from sglang.srt.utils.tensor_bridge import _export_evaluated_mlx

    return _export_evaluated_mlx(array, torch.device("mps"), _mlx()).float()


def _to_mlx(tensor):
    from sglang.srt.utils.tensor_bridge import torch_to_mlx

    torch.mps.synchronize()
    return torch_to_mlx(tensor)


class TestGdnOps(CustomTestCase):
    def _state(self, layers=3, slots=9, dtype=torch.bfloat16):
        return (
            torch.randn(layers, slots, CHANNELS, KERNEL - 1, dtype=dtype, device="mps"),
            torch.randn(
                layers, slots, NUM_V_HEADS, HEAD_V_DIM, HEAD_K_DIM, device="mps"
            )
            * 0.1,
            torch.tensor([0, 4, 7, 2], dtype=torch.int32, device="mps"),
        )

    def test_decode_matches_reference_over_pool_and_pending_rows(self):
        from sglang.kernels.ops.attention.mlx.gated_delta import gdn_decode as mlx_op
        from sglang.kernels.ops.attention.mlx.gated_delta_export import gdn_decode

        torch.manual_seed(0)
        conv_pool, ssm_pool, table = self._state()
        weight = torch.randn(CHANNELS, KERNEL, dtype=torch.bfloat16, device="mps") * 0.5
        A_log = torch.randn(NUM_V_HEADS, device="mps") - 2
        dt_bias = torch.randn(NUM_V_HEADS, dtype=torch.bfloat16, device="mps")
        for batch, requests in ((1, [2]), (3, [1, 3, 2])):
            query_key_value = torch.randn(
                batch, CHANNELS, dtype=torch.bfloat16, device="mps"
            )
            a = torch.randn(batch, NUM_V_HEADS, dtype=torch.bfloat16, device="mps")
            b = torch.randn_like(a)
            for tail in (False, True):
                pending = {}
                if tail:
                    pending = dict(
                        tail_conv=torch.randn(
                            batch,
                            CHANNELS,
                            KERNEL - 1,
                            dtype=torch.bfloat16,
                            device="mps",
                        ),
                        tail_ssm=torch.randn(
                            batch,
                            NUM_V_HEADS,
                            HEAD_V_DIM,
                            HEAD_K_DIM,
                            device="mps",
                        )
                        * 0.1,
                    )
                expected = gdn_decode(
                    query_key_value,
                    a,
                    b,
                    weight,
                    None,
                    A_log,
                    dt_bias,
                    conv_pool,
                    ssm_pool,
                    table,
                    torch.tensor(requests, device="mps"),
                    layer=1,
                    **pending,
                    **GEOMETRY,
                )
                actual = mlx_op(
                    _to_mlx(query_key_value),
                    _to_mlx(a),
                    _to_mlx(b),
                    _to_mlx(weight),
                    None,
                    _to_mlx(A_log),
                    _to_mlx(dt_bias),
                    _to_mlx(conv_pool),
                    _to_mlx(ssm_pool),
                    _to_mlx(table),
                    _to_mlx(torch.tensor(requests, device="mps")),
                    layer=1,
                    tails=(
                        (_to_mlx(pending["tail_conv"]), _to_mlx(pending["tail_ssm"]))
                        if tail
                        else None
                    ),
                    **GEOMETRY,
                )
                _mlx().eval(*actual)
                for name, got, want in zip(("out", "conv", "ssm"), actual, expected):
                    with self.subTest(batch=batch, tail=tail, output=name):
                        torch.testing.assert_close(
                            _to_torch(got), want.float(), rtol=1e-2, atol=1e-4
                        )

    def test_prefill_matches_reference_from_a_zero_state(self):
        from sglang.kernels.ops.attention.mlx.gated_delta import gdn_prefill as mlx_op
        from sglang.kernels.ops.attention.mlx.gated_delta_export import gdn_prefill

        torch.manual_seed(1)
        weight = torch.randn(CHANNELS, KERNEL, dtype=torch.bfloat16, device="mps") * 0.5
        A_log = torch.randn(NUM_V_HEADS, device="mps") - 2
        dt_bias = torch.randn(NUM_V_HEADS, dtype=torch.bfloat16, device="mps")
        # T below the kernel width exercises the short final window.
        for tokens in (1, 2, 3, 5, 37):
            query_key_value = torch.randn(
                tokens, CHANNELS, dtype=torch.bfloat16, device="mps"
            )
            a = torch.randn(tokens, NUM_V_HEADS, dtype=torch.bfloat16, device="mps")
            b = torch.randn_like(a)
            expected = gdn_prefill(
                query_key_value,
                a,
                b,
                weight,
                None,
                A_log,
                dt_bias,
                conv_dtype=torch.bfloat16,
                ssm_dtype=torch.float32,
                **GEOMETRY,
            )
            actual = mlx_op(
                _to_mlx(query_key_value),
                _to_mlx(a),
                _to_mlx(b),
                _to_mlx(weight),
                None,
                _to_mlx(A_log),
                _to_mlx(dt_bias),
                conv_dtype=torch.bfloat16,
                ssm_dtype=torch.float32,
                **GEOMETRY,
            )
            _mlx().eval(*actual)
            for name, got, want in zip(("out", "conv", "ssm"), actual, expected):
                with self.subTest(tokens=tokens, output=name):
                    torch.testing.assert_close(
                        _to_torch(got), want.float(), rtol=1e-2, atol=1e-4
                    )

    def test_fp32_recurrence_does_not_drift_over_512_steps(self):
        """bf16 decay rounds at every step (ml-explore/mlx#4760); fp32 must not."""
        from sglang.kernels.ops.attention.mlx.gated_delta import gdn_decode as mlx_op
        from sglang.kernels.ops.attention.mlx.gated_delta_export import gdn_decode

        torch.manual_seed(2)
        dtype = torch.bfloat16
        conv_pool, ssm_pool, table = self._state(layers=1, slots=3)
        weight = torch.randn(CHANNELS, KERNEL, dtype=dtype, device="mps") * 0.5
        A_log = torch.randn(NUM_V_HEADS, device="mps") - 3
        dt_bias = torch.randn(NUM_V_HEADS, dtype=dtype, device="mps")
        requests = torch.tensor([1], device="mps")
        conv_row, ssm_row = conv_pool[0, 1][None], ssm_pool[0, 1][None]
        reference_conv, reference_ssm = conv_row.clone(), ssm_row.clone()
        for step in range(512):
            query_key_value = torch.randn(1, CHANNELS, dtype=dtype, device="mps")
            a = torch.randn(1, NUM_V_HEADS, dtype=dtype, device="mps")
            b = torch.randn_like(a)
            actual = mlx_op(
                _to_mlx(query_key_value),
                _to_mlx(a),
                _to_mlx(b),
                _to_mlx(weight),
                None,
                _to_mlx(A_log),
                _to_mlx(dt_bias),
                _to_mlx(conv_pool),
                _to_mlx(ssm_pool),
                _to_mlx(table),
                _to_mlx(requests),
                layer=0,
                tails=(_to_mlx(conv_row), _to_mlx(ssm_row)),
                **GEOMETRY,
            )
            _mlx().eval(*actual)
            conv_row, ssm_row = _to_torch(actual[1]), _to_torch(actual[2])
            reference_conv, reference_ssm = gdn_decode(
                query_key_value,
                a,
                b,
                weight,
                None,
                A_log,
                dt_bias,
                conv_pool,
                ssm_pool,
                table,
                requests,
                tail_conv=reference_conv,
                tail_ssm=reference_ssm,
                layer=0,
                **GEOMETRY,
            )[1:]
        # The fp32 reference recurrence and MLX agree to fp32 noise, not to the
        # several-percent drift of a decay rounded to bf16.
        torch.testing.assert_close(ssm_row, reference_ssm, rtol=1e-3, atol=1e-4)
        torch.testing.assert_close(
            conv_row.float(), reference_conv.float(), rtol=0, atol=0
        )


class _ToyGdn(torch.nn.Module):
    """One GDN mixer over its own conv weight and state parameters."""

    def __init__(self, hidden, layer_id):
        super().__init__()
        self.in_qkv = torch.nn.Linear(hidden, CHANNELS, bias=False, device="mps")
        self.in_ba = torch.nn.Linear(hidden, 2 * NUM_V_HEADS, bias=False, device="mps")
        self.conv1d = torch.nn.Module()
        self.conv1d.weight = torch.nn.Parameter(
            torch.randn(CHANNELS, 1, KERNEL, device="mps") * 0.5
        )
        self.A_log = torch.nn.Parameter(torch.randn(NUM_V_HEADS, device="mps") - 2)
        self.dt_bias = torch.nn.Parameter(torch.randn(NUM_V_HEADS, device="mps"))
        self.attn = _ToyRegion.linear(
            layer_id, self.conv1d.weight, self.A_log, self.dt_bias
        )
        self.out = torch.nn.Linear(
            NUM_V_HEADS * HEAD_V_DIM, hidden, bias=False, device="mps"
        )

    def forward(self, hidden, forward_batch):
        mixed = self.in_qkv(hidden)
        b, a = self.in_ba(hidden).split((NUM_V_HEADS, NUM_V_HEADS), dim=-1)
        core = self.attn(forward_batch, mixed_qkv=mixed, a=a, b=b)
        return self.out(core.reshape(hidden.shape[0], -1))


class _ToyAttention(torch.nn.Module):
    def __init__(self, hidden, layer_id):
        super().__init__()
        self.qkv = torch.nn.Linear(
            hidden, (4 + 2) * HEAD_K_DIM, bias=False, device="mps"
        )
        self.attn = _ToyRegion.attention(4, layer_id)
        self.out = torch.nn.Linear(4 * HEAD_K_DIM, hidden, bias=False, device="mps")

    def forward(self, hidden, forward_batch):
        q, k, v = self.qkv(hidden).split(
            [4 * HEAD_K_DIM, HEAD_K_DIM, HEAD_K_DIM], dim=-1
        )
        return self.out(self.attn(q, k, v, forward_batch))


class _ToyRegion:
    """Factory helpers so the toy layers import RadixAttention lazily."""

    @staticmethod
    def linear(layer_id, weight, A_log, dt_bias):
        from sglang.srt.layers.radix_linear_attention import RadixLinearAttention

        return RadixLinearAttention(
            layer_id=layer_id,
            num_q_heads=NUM_K_HEADS,
            num_k_heads=NUM_K_HEADS,
            num_v_heads=NUM_V_HEADS,
            head_q_dim=HEAD_K_DIM,
            head_k_dim=HEAD_K_DIM,
            head_v_dim=HEAD_V_DIM,
            conv_weights=weight.view(CHANNELS, KERNEL),
            bias=None,
            activation="silu",
            A_log=A_log,
            dt_bias=dt_bias,
        )

    @staticmethod
    def attention(heads, layer_id):
        from sglang.srt.layers.radix_attention import RadixAttention

        return RadixAttention(heads, HEAD_K_DIM, HEAD_K_DIM**-0.5, 1, layer_id)


class _ToyHybrid(torch.nn.Module):
    def __init__(self, hidden=32, vocab=97):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab)
        self.embed = torch.nn.Embedding(vocab, hidden, device="mps")
        self.layers = torch.nn.ModuleList(
            [_ToyGdn(hidden, 0), _ToyAttention(hidden, 1), _ToyGdn(hidden, 2)]
        )
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False, device="mps")

    def forward(self, input_ids, positions, forward_batch):
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = hidden + layer(hidden, forward_batch)
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        logits = self.lm_head(hidden).float()
        if forward_batch.forward_mode.is_extend():
            logits = logits[-1:]
        return LogitsProcessorOutput(next_token_logits=logits)


class TestHybridGdnRegion(CustomTestCase):
    """A hybrid stack exports and replays through the generic stateful ops."""

    def _pools(self, layers=2, slots=8, dtype=torch.bfloat16):
        return dict(
            table=torch.zeros(4, 32, device="mps", dtype=torch.int32),
            state_table=torch.tensor([0, 3, 5, 0], dtype=torch.int32, device="mps"),
            conv=torch.randn(
                layers, slots, CHANNELS, KERNEL - 1, dtype=dtype, device="mps"
            ),
            ssm=torch.randn(
                layers, slots, NUM_V_HEADS, HEAD_V_DIM, HEAD_K_DIM, device="mps"
            )
            * 0.1,
        )

    @staticmethod
    def _state(pools):
        return SimpleNamespace(
            table=pools["state_table"],
            conv=pools["conv"],
            ssm=pools["ssm"],
        )

    def _wrapper(self, model, region, pools):
        from sglang.srt.hardware_backend.mps.compiled_runner import _DecodeModule

        cache_source = SimpleNamespace(req_to_token=pools["table"])
        kv_source = SimpleNamespace(
            get_kv_buffer=lambda _: (pools["k"], pools["v"]), page_size=1
        )
        return _DecodeModule(
            model=model,
            req_pool=cache_source,
            kv_pool=kv_source,
            region=region,
            state=self._state(pools),
        ).eval()

    def test_region_admits_gdn_blocks_and_rejects_other_linear_attention(self):
        from sglang.srt.hardware_backend.mps.compiled_region import (
            UnsupportedMlxRegion,
            discover_decode_region,
        )

        model = _ToyHybrid()
        region = discover_decode_region(model)
        self.assertEqual(tuple(x.layer_id for x in region.layers), (1,))
        self.assertEqual(tuple(x.layer_id for x in region.linear_layers), (0, 2))
        model.forward = lambda input_ids, positions, forward_batch, **kwargs: None
        discover_decode_region(model)
        # A KDA-style layer has no conv weights and must stay on eager MPS.
        model.layers[0].attn.conv_weights = None
        with self.assertRaisesRegex(UnsupportedMlxRegion, "GDN geometry"):
            discover_decode_region(model)

    def test_commit_writes_only_selected_rows_per_state_layer(self):
        """A layer-indexing error must not corrupt another recurrent state row."""
        from sglang.srt.hardware_backend.mps.compiled_state import commit_gdn_state

        pools = self._pools(layers=2, dtype=torch.float32)
        state = SimpleNamespace(
            table=pools["state_table"],
            conv=pools["conv"],
            ssm=pools["ssm"],
        )
        slots = torch.tensor([5, 2], device="mps")
        conv_rows = torch.randn(2, 2, CHANNELS, KERNEL - 1, device="mps")
        ssm_rows = [
            torch.randn(2, NUM_V_HEADS, HEAD_V_DIM, HEAD_K_DIM, device="mps")
            for _ in range(2)
        ]
        expected_conv = pools["conv"].clone()
        expected_ssm = pools["ssm"].clone()
        expected_conv[:, slots] = conv_rows
        for layer, rows in enumerate(ssm_rows):
            expected_ssm[layer].index_copy_(0, slots, rows)

        commit_gdn_state(state, slots=slots, conv_rows=conv_rows, ssm_rows=ssm_rows)

        torch.testing.assert_close(pools["conv"], expected_conv, rtol=0, atol=0)
        torch.testing.assert_close(pools["ssm"], expected_ssm, rtol=0, atol=0)

    def test_runner_commits_complete_prefill_rows(self):
        """The compiled B1 prefill must commit every value to its request slot."""
        from sglang.srt.hardware_backend.mps.compiled_runner import CompiledMlxRunner

        pools = self._pools(layers=2, dtype=torch.float32)
        runner = object.__new__(CompiledMlxRunner)
        runner._prefill = True
        runner._state = self._state(pools)
        batch = SimpleNamespace(req_pool_indices=torch.tensor([1], device="mps"))
        conv_rows = torch.randn(2, 1, CHANNELS, KERNEL - 1, device="mps")
        ssm_rows = tuple(
            torch.randn(1, NUM_V_HEADS, HEAD_V_DIM, HEAD_K_DIM, device="mps")
            for _ in range(2)
        )
        expected_conv = pools["conv"].clone()
        expected_ssm = pools["ssm"].clone()
        slots = pools["state_table"][[1]].long()
        expected_conv[:, slots] = conv_rows
        for layer, rows in enumerate(ssm_rows):
            expected_ssm[layer].index_copy_(0, slots, rows)

        runner._commit_state(batch, conv_rows, *ssm_rows)

        torch.testing.assert_close(pools["conv"], expected_conv, rtol=0, atol=0)
        torch.testing.assert_close(pools["ssm"], expected_ssm, rtol=0, atol=0)

    def test_hybrid_lookahead_state_slot_change_discards_pending_outputs(self):
        """A new state slot must discard lookahead without writing either pool."""
        import mlx.core as mx

        from sglang.srt.hardware_backend.mps.compiled_graph import CompiledMlxGraph
        from sglang.srt.hardware_backend.mps.compiled_region import (
            discover_decode_region,
        )
        from sglang.srt.hardware_backend.mps.compiled_runner import (
            CompiledMlxRunner,
            _batch_inputs,
        )
        from sglang.srt.hardware_backend.mps.compiled_state import stateful_ops
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )
        from sglang.srt.model_executor.forward_mode import ForwardMode
        from sglang.srt.runtime_context import get_context

        torch.manual_seed(0)
        model = _ToyHybrid().eval()
        region = discover_decode_region(model)
        pools = self._pools(layers=2, dtype=torch.float32)
        pools["k"] = torch.randn(64, 1, HEAD_K_DIM, device="mps")
        pools["v"] = torch.randn(64, 1, HEAD_K_DIM, device="mps")
        pools["table"][1, :4] = torch.arange(10, 14, dtype=torch.int32)
        batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=1,
            input_ids=torch.tensor([7], device="mps"),
            positions=torch.tensor([1], device="mps"),
            req_pool_indices=torch.tensor([1], device="mps"),
            seq_lens=torch.tensor([2], device="mps"),
            seq_lens_sum=2,
            out_cache_loc=torch.tensor([11], device="mps"),
        )
        batch.sampling_info = SimpleNamespace(
            is_all_greedy=True,
            grammars=None,
            grammar_mask=None,
            has_custom_logit_processor=False,
            logit_bias=None,
            acc_additive_penalties=None,
            acc_scaling_penalties=None,
            penalizer_orchestrator=None,
        )
        wrapper = self._wrapper(model, region, pools)
        with forward_context(ForwardContext(attn_backend=wrapper.attention)):
            graph = CompiledMlxGraph(
                model=wrapper,
                example_inputs=_batch_inputs(batch),
                stateful=stateful_ops(region),
            )
        self.addCleanup(graph.close)

        def set_kv_buffer(layer, locations, keys, values):
            pools["k"][locations] = keys
            pools["v"][locations] = values

        runner = object.__new__(CompiledMlxRunner)
        runner._prefill = False
        runner._layers = region.layers
        runner._linear_layers = region.linear_layers
        runner._state = self._state(pools)
        runner.model_runner = SimpleNamespace(
            model=model,
            model_config=SimpleNamespace(context_len=32),
            req_to_token_pool=SimpleNamespace(
                req_to_token=pools["table"],
                req_generation=torch.zeros(4, dtype=torch.int64),
            ),
            token_to_kv_pool=SimpleNamespace(
                page_size=1,
                get_kv_buffer=lambda _: (pools["k"], pools["v"]),
                get_key_buffer=lambda _: pools["k"],
                set_kv_buffer=set_kv_buffer,
            ),
        )
        runner._max_batch_size = 4
        runner._unsupported_reason = None
        runner._unsupported_batches = {}
        runner._model_structure = (
            model,
            tuple(
                (module, tuple(module._modules.items())) for module in model.modules()
            ),
        )
        runner._weight_sources = ()
        runner._fallback_reasons = set()
        runner.execution_count = 0
        with get_context().override_server_args(device="mps"):
            runner._init_backend()
        with get_context().override_server_args(disable_overlap_schedule=True):
            runner.enable_lookahead()
        commit_state = runner._commit_state
        pools_at_commit = []

        def observed_commit(forward_batch, *rows):
            pools_at_commit.append((pools["conv"].clone(), pools["ssm"].clone()))
            commit_state(forward_batch, *rows)

        def next_step(location):
            batch.input_ids = torch.tensor(
                runner._pending.tokens.tolist(), device="mps"
            )
            batch.positions = batch.positions + 1
            batch.seq_lens = batch.seq_lens + 1
            batch.out_cache_loc = torch.tensor([location], device="mps")
            torch.mps.synchronize()

        with (
            patch.object(runner, "_graph", return_value=graph),
            patch.object(runner, "_commit_state", observed_commit),
        ):
            runner.execute(batch)
            self.assertIsNotNone(runner._pending)
            before = (pools["conv"].clone(), pools["ssm"].clone())
            pools["state_table"][1] = 4
            next_step(12)
            runner.execute(batch)
            self.assertEqual(runner.prefetch_discard_reasons, {"metadata": 1})
            self.assertEqual(runner.prefetch_hits, 0)
            # No discarded row reaches a pool before the replacement step commits.
            torch.testing.assert_close(pools_at_commit[-1], before, rtol=0, atol=0)
            # An unchanged slot hits and commits the rows a synchronous step computes.
            next_step(13)
            expected = graph.launch(graph.bind(_batch_inputs(batch)))
            mx.eval(*expected)
            runner.execute(batch)
            self.assertEqual(runner.prefetch_hits, 1)
            torch.testing.assert_close(
                pools["conv"][:, 4], _to_torch(expected[3])[:, 0], rtol=0, atol=0
            )
            torch.testing.assert_close(
                pools["ssm"][:, 4],
                torch.cat([_to_torch(rows) for rows in expected[4:]]),
                rtol=0,
                atol=0,
            )
        runner.drain()

    def test_whole_forward_state_and_lookahead_tails(self):
        import mlx.core as mx

        from sglang.srt.hardware_backend.mps.compiled_graph import CompiledMlxGraph
        from sglang.srt.hardware_backend.mps.compiled_region import (
            discover_decode_region,
        )
        from sglang.srt.hardware_backend.mps.compiled_state import stateful_ops
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )

        torch.manual_seed(0)
        model = _ToyHybrid().eval()
        region = discover_decode_region(model)
        state_layers = 2
        pools = self._pools(layers=state_layers)
        pools["k"] = torch.randn(64, 1, HEAD_K_DIM, device="mps")
        pools["v"] = torch.randn(64, 1, HEAD_K_DIM, device="mps")
        pools["table"][1, :10] = torch.arange(10, 20, dtype=torch.int32)
        pools["table"][2, :12] = torch.arange(30, 42, dtype=torch.int32)
        lengths = torch.tensor([5, 7], device="mps")
        inputs = (
            torch.tensor([7, 11], device="mps"),
            lengths - 1,
            torch.tensor([1, 2], device="mps"),
            lengths,
            torch.stack([pools["table"][1, 4], pools["table"][2, 6]]).long(),
        )
        wrapper = self._wrapper(model, region, pools)
        with (
            torch.no_grad(),
            forward_context(ForwardContext(attn_backend=wrapper.attention)),
        ):
            reference = wrapper(*inputs)
            graph = CompiledMlxGraph(
                model=wrapper,
                example_inputs=inputs,
                stateful=stateful_ops(region),
            )
            self.addCleanup(graph.close)
            self.assertEqual(len(graph.stateful), 2)
            torch.mps.synchronize()
            outputs = graph.launch(graph.bind(inputs))
            mx.eval(*outputs)
            for name, got, want in zip(
                ("logits", "keys", "values", "conv", "ssm1", "ssm2"), outputs, reference
            ):
                with self.subTest(output=name):
                    torch.testing.assert_close(
                        _to_torch(got), want.float(), rtol=2e-2, atol=2e-2
                    )
            # Lookahead: the next step reads pending rows, not the pool.
            from sglang.srt.hardware_backend.mps.compiled_runner import (
                _following_inputs,
            )

            ahead = graph.launch(
                _following_inputs(
                    graph.bindings,
                    graph.bind(inputs),
                    mx.argmax(outputs[0], axis=-1).astype(mx.int64),
                ),
                tails=outputs[1:],
            )
            mx.eval(*ahead)
            slots = pools["state_table"][[1, 2]].long()
            committed = {
                "conv": pools["conv"].clone(),
                "ssm": pools["ssm"].clone(),
            }
            committed["conv"][:, slots] = _to_torch(outputs[3]).to(
                committed["conv"].dtype
            )
            for index in range(state_layers):
                committed["ssm"][index].index_copy_(
                    0, slots, _to_torch(outputs[4 + index]).to(committed["ssm"].dtype)
                )
            pools["conv"].copy_(committed["conv"])
            pools["ssm"].copy_(committed["ssm"])
            keys, values = _to_torch(outputs[1]), _to_torch(outputs[2])
            for request, location in enumerate(inputs[4]):
                pools["k"][location.long()] = keys[0][request].to(pools["k"].dtype)
                pools["v"][location.long()] = values[0][request].to(pools["v"].dtype)
            next_inputs = (
                _to_torch(mx.argmax(outputs[0], axis=-1).astype(mx.int64)).long(),
                inputs[1] + 1,
                inputs[2],
                lengths + 1,
                torch.stack([pools["table"][1, 5], pools["table"][2, 7]]).long(),
            )
            torch.mps.synchronize()
            committed_outputs = graph.launch(graph.bind(next_inputs))
            mx.eval(*committed_outputs)
            for name, got, want in zip(
                ("logits", "keys", "values", "conv", "ssm1", "ssm2"),
                ahead,
                committed_outputs,
            ):
                with self.subTest(output=name, step="lookahead"):
                    torch.testing.assert_close(
                        _to_torch(got), _to_torch(want), rtol=0, atol=0
                    )

    def test_compiled_prefill_runs_gdn_from_a_zero_state(self):
        import mlx.core as mx

        from sglang.kernels.ops.attention.mlx.gated_delta import (
            gdn_prefill as mlx_gdn_prefill,
        )
        from sglang.kernels.ops.attention.mlx.gated_delta_export import gdn_prefill
        from sglang.srt.hardware_backend.mps.compiled_graph import CompiledMlxGraph
        from sglang.srt.hardware_backend.mps.compiled_region import (
            discover_decode_region,
        )
        from sglang.srt.hardware_backend.mps.compiled_runner import (
            _causal_sdpa,
            _DecodeModule,
        )
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )

        torch.manual_seed(3)
        model = _ToyHybrid().eval()
        region = discover_decode_region(model)
        pools = self._pools(layers=2)
        pools["k"] = torch.randn(64, 1, HEAD_K_DIM, device="mps")
        pools["v"] = torch.randn(64, 1, HEAD_K_DIM, device="mps")
        tokens = 5
        pools["table"][1, :tokens] = torch.arange(1, tokens + 1, dtype=torch.int32)
        inputs = (
            torch.tensor([7, 11, 3, 5, 9], device="mps"),
            torch.arange(tokens, device="mps"),
            torch.tensor([1], device="mps"),
            torch.tensor([tokens], device="mps"),
            torch.arange(1, tokens + 1, device="mps"),
        )
        wrapper = _DecodeModule(
            model=model,
            req_pool=SimpleNamespace(req_to_token=pools["table"]),
            kv_pool=SimpleNamespace(
                get_kv_buffer=lambda _: (pools["k"], pools["v"]), page_size=1
            ),
            region=region,
            state=self._state(pools),
            prefill=True,
        ).eval()
        lowerings = {
            torch.ops.aten.scaled_dot_product_attention.default: _causal_sdpa,
            gdn_prefill._opoverload: mlx_gdn_prefill,
        }
        with (
            torch.no_grad(),
            forward_context(ForwardContext(attn_backend=wrapper.attention)),
        ):
            # T=2 forces the dynamic-token path: no shape is baked into the graph.
            graph = CompiledMlxGraph(
                model=wrapper,
                example_inputs=(
                    inputs[0][:2],
                    inputs[1][:2],
                    inputs[2],
                    torch.tensor([2], device="mps"),
                    inputs[4][:2],
                ),
                custom_lowerings=lowerings,
                dynamic_tokens=True,
            )
            self.addCleanup(graph.close)
            for count in (2, tokens):
                current = (
                    inputs[0][:count],
                    inputs[1][:count],
                    inputs[2],
                    torch.tensor([count], device="mps"),
                    inputs[4][:count],
                )
                with torch.no_grad():
                    expected = wrapper(*current)
                torch.mps.synchronize()
                outputs = graph.launch(graph.bind(current))
                mx.eval(*outputs)
                for name, got, want in zip(
                    ("logits", "keys", "values", "conv", "ssm1", "ssm2"),
                    outputs,
                    expected,
                ):
                    with self.subTest(tokens=count, output=name):
                        torch.testing.assert_close(
                            _to_torch(got), want.float(), rtol=2e-2, atol=2e-2
                        )


if __name__ == "__main__":
    unittest.main()
