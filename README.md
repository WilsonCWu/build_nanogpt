# build_nanogpt
following along https://www.youtube.com/watch?v=l8pRSuU81PU


# misc notes from video
* when trying to overfit on tinyshakespeare, we expect to see big loss improvement from first few batches because most of the 50k tokens aren't being used, and it can immediately drive most of those to 0
* GPT3 gradually increases batch size linearly initially. intuition is that initially, you're just learning what tokens are used frequently, and for that basic task, different batches are extremely correlated. no point of larger batches initially
* data sampled without replacement per epoch



# errors
compile + eval erro
```

[rank2]: Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information


[rank2]: You can suppress this exception and fall back to eager by setting:
[rank2]:     import torch._dynamo
[rank2]:     torch._dynamo.config.suppress_errors = True

[rank5]: Traceback (most recent call last):
[rank5]:   File "/home/ubuntu/nanogpt-train/build_nanogpt/train_gpt2.py", line 444, in <module>
[rank5]:     logits, loss = model(tokens)
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
[rank5]:     return self._call_impl(*args, **kwargs)
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
[rank5]:     return forward_call(*args, **kwargs)
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/parallel/distributed.py", line 1593, in forward
[rank5]:     else self._run_ddp_forward(*inputs, **kwargs)
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/parallel/distributed.py", line 1411, in _run_ddp_forward
[rank5]:     return self.module(*inputs, **kwargs)  # type: ignore[index]
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
[rank5]:     return self._call_impl(*args, **kwargs)
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
[rank5]:     return forward_call(*args, **kwargs)
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py", line 451, in _fn
[rank5]:     return fn(*args, **kwargs)
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
[rank5]:     return self._call_impl(*args, **kwargs)
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
[rank5]:     return forward_call(*args, **kwargs)
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py", line 917, in catch_errors
[rank5]:     return hijacked_callback(frame, cache_entry, hooks, frame_state)
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py", line 786, in _convert_frame
[rank5]:     result = inner_convert(
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py", line 400, in _convert_frame_assert
[rank5]:     return _compile(
[rank5]:   File "/usr/lib/python3.10/contextlib.py", line 79, in inner
[rank5]:     return func(*args, **kwds)
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py", line 676, in _compile
[rank5]:     guarded_code = compile_inner(code, one_graph, hooks, transform)
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/utils.py", line 262, in time_wrapper
[rank5]:     r = func(*args, **kwargs)
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py", line 535, in compile_inner
[rank5]:     out_code = transform_code_object(code, transform)
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/bytecode_transformation.py", line 1036, in transform_code_object
[rank5]:     transformations(instructions, code_options)
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py", line 165, in _fn
[rank5]:     return fn(*args, **kwargs)
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py", line 500, in transform
[rank5]:     tracer.run()
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py", line 2149, in run
[rank5]:     super().run()
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py", line 810, in run
[rank5]:     and self.step()
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py", line 773, in step
[rank5]:     getattr(self, inst.opname)(inst)
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py", line 2268, in RETURN_VALUE
[rank5]:     self.output.compile_subgraph(
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/output_graph.py", line 1001, in compile_subgraph
[rank5]:     self.compile_and_call_fx_graph(tx, pass2.graph_output_vars(), root)
[rank5]:   File "/usr/lib/python3.10/contextlib.py", line 79, in inner
[rank5]:     return func(*args, **kwds)
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/output_graph.py", line 1178, in compile_and_call_fx_graph
[rank5]:     compiled_fn = self.call_user_compiler(gm)
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/utils.py", line 262, in time_wrapper
[rank5]:     r = func(*args, **kwargs)
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/output_graph.py", line 1251, in call_user_compiler
[rank5]:     raise BackendCompilerFailed(self.compiler_fn, e).with_traceback(
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/output_graph.py", line 1232, in call_user_compiler
[rank5]:     compiled_fn = compiler_fn(gm, self.example_inputs())
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/backends/distributed.py", line 606, in compile_fn
[rank5]:     submod_compiler.run(*example_inputs)
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/fx/interpreter.py", line 145, in run
[rank5]:     self.env[node] = self.run_node(node)
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/backends/distributed.py", line 348, in run_node
[rank5]:     compiled_submod_real = self.compile_submod(real_mod, new_args, kwargs)
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/backends/distributed.py", line 263, in compile_submod
[rank5]:     self.compiler(input_mod, args),
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/repro/after_dynamo.py", line 117, in debug_wrapper
[rank5]:     compiled_gm = compiler_fn(gm, example_inputs)
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/__init__.py", line 1731, in __call__
[rank5]:     return compile_fx(model_, inputs_, config_patches=self.config)
[rank5]:   File "/usr/lib/python3.10/contextlib.py", line 79, in inner
[rank5]:     return func(*args, **kwds)
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_inductor/compile_fx.py", line 1330, in compile_fx
[rank5]:     return aot_autograd(
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/backends/common.py", line 58, in compiler_fn
[rank5]:     cg = aot_module_simplified(gm, example_inputs, **kwargs)
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_functorch/aot_autograd.py", line 903, in aot_module_simplified
[rank5]:     compiled_fn = create_aot_dispatcher_function(
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/utils.py", line 262, in time_wrapper
[rank5]:     r = func(*args, **kwargs)
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_functorch/aot_autograd.py", line 628, in create_aot_dispatcher_function
[rank5]:     compiled_fn = compiler_fn(flat_fn, fake_flat_args, aot_config, fw_metadata=fw_metadata)
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py", line 443, in aot_wrapper_dedupe
[rank5]:     return compiler_fn(flat_fn, leaf_flat_args, aot_config, fw_metadata=fw_metadata)
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py", line 648, in aot_wrapper_synthetic_base
[rank5]:     return compiler_fn(flat_fn, flat_args, aot_config, fw_metadata=fw_metadata)
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py", line 123, in aot_dispatch_base
[rank5]:     fakified_out = _compute_output_meta_with_inductor_strides(
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py", line 70, in _compute_output_meta_with_inductor_strides
[rank5]:     out = [n.meta["val"] for n in (list(fw_module.graph.nodes)[-1].args[0])]
[rank5]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py", line 70, in <listcomp>
[rank5]:     out = [n.meta["val"] for n in (list(fw_module.graph.nodes)[-1].args[0])]
[rank5]: torch._dynamo.exc.BackendCompilerFailed: backend='compile_fn' raised:
[rank5]: AttributeError: 'int' object has no attribute 'meta'

[rank5]: While executing %submod_2 : [num_users=4] = call_module[target=submod_2](args = (%getitem, %s0, %s1, %getitem_1, %getitem_2, %getitem_3), kwargs = {})
[rank5]: Original traceback:
[rank5]: None

[rank5]: Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information


[rank5]: You can suppress this exception and fall back to eager by setting:
[rank5]:     import torch._dynamo
[rank5]:     torch._dynamo.config.suppress_errors = True

[rank3]: Traceback (most recent call last):
[rank3]:   File "/home/ubuntu/nanogpt-train/build_nanogpt/train_gpt2.py", line 444, in <module>
[rank3]:     logits, loss = model(tokens)
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
[rank3]:     return self._call_impl(*args, **kwargs)
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
[rank3]:     return forward_call(*args, **kwargs)
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/parallel/distributed.py", line 1593, in forward
[rank3]:     else self._run_ddp_forward(*inputs, **kwargs)
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/parallel/distributed.py", line 1411, in _run_ddp_forward
[rank3]:     return self.module(*inputs, **kwargs)  # type: ignore[index]
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
[rank3]:     return self._call_impl(*args, **kwargs)
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
[rank3]:     return forward_call(*args, **kwargs)
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py", line 451, in _fn
[rank3]:     return fn(*args, **kwargs)
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
[rank3]:     return self._call_impl(*args, **kwargs)
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
[rank3]:     return forward_call(*args, **kwargs)
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py", line 917, in catch_errors
[rank3]:     return hijacked_callback(frame, cache_entry, hooks, frame_state)
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py", line 786, in _convert_frame
[rank3]:     result = inner_convert(
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py", line 400, in _convert_frame_assert
[rank3]:     return _compile(
[rank3]:   File "/usr/lib/python3.10/contextlib.py", line 79, in inner
[rank3]:     return func(*args, **kwds)
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py", line 676, in _compile
[rank3]:     guarded_code = compile_inner(code, one_graph, hooks, transform)
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/utils.py", line 262, in time_wrapper
[rank3]:     r = func(*args, **kwargs)
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py", line 535, in compile_inner
[rank3]:     out_code = transform_code_object(code, transform)
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/bytecode_transformation.py", line 1036, in transform_code_object
[rank3]:     transformations(instructions, code_options)
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py", line 165, in _fn
[rank3]:     return fn(*args, **kwargs)
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py", line 500, in transform
[rank3]:     tracer.run()
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py", line 2149, in run
[rank3]:     super().run()
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py", line 810, in run
[rank3]:     and self.step()
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py", line 773, in step
[rank3]:     getattr(self, inst.opname)(inst)
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py", line 2268, in RETURN_VALUE
[rank3]:     self.output.compile_subgraph(
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/output_graph.py", line 1001, in compile_subgraph
[rank3]:     self.compile_and_call_fx_graph(tx, pass2.graph_output_vars(), root)
[rank3]:   File "/usr/lib/python3.10/contextlib.py", line 79, in inner
[rank3]:     return func(*args, **kwds)
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/output_graph.py", line 1178, in compile_and_call_fx_graph
[rank3]:     compiled_fn = self.call_user_compiler(gm)
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/utils.py", line 262, in time_wrapper
[rank3]:     r = func(*args, **kwargs)
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/output_graph.py", line 1251, in call_user_compiler
[rank3]:     raise BackendCompilerFailed(self.compiler_fn, e).with_traceback(
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/output_graph.py", line 1232, in call_user_compiler
[rank3]:     compiled_fn = compiler_fn(gm, self.example_inputs())
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/backends/distributed.py", line 606, in compile_fn
[rank3]:     submod_compiler.run(*example_inputs)
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/fx/interpreter.py", line 145, in run
[rank3]:     self.env[node] = self.run_node(node)
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/backends/distributed.py", line 348, in run_node
[rank3]:     compiled_submod_real = self.compile_submod(real_mod, new_args, kwargs)
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/backends/distributed.py", line 263, in compile_submod
[rank3]:     self.compiler(input_mod, args),
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/repro/after_dynamo.py", line 117, in debug_wrapper
[rank3]:     compiled_gm = compiler_fn(gm, example_inputs)
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/__init__.py", line 1731, in __call__
[rank3]:     return compile_fx(model_, inputs_, config_patches=self.config)
[rank3]:   File "/usr/lib/python3.10/contextlib.py", line 79, in inner
[rank3]:     return func(*args, **kwds)
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_inductor/compile_fx.py", line 1330, in compile_fx
[rank3]:     return aot_autograd(
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/backends/common.py", line 58, in compiler_fn
[rank3]:     cg = aot_module_simplified(gm, example_inputs, **kwargs)
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_functorch/aot_autograd.py", line 903, in aot_module_simplified
[rank3]:     compiled_fn = create_aot_dispatcher_function(
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/utils.py", line 262, in time_wrapper
[rank3]:     r = func(*args, **kwargs)
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_functorch/aot_autograd.py", line 628, in create_aot_dispatcher_function
[rank3]:     compiled_fn = compiler_fn(flat_fn, fake_flat_args, aot_config, fw_metadata=fw_metadata)
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py", line 443, in aot_wrapper_dedupe
[rank3]:     return compiler_fn(flat_fn, leaf_flat_args, aot_config, fw_metadata=fw_metadata)
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py", line 648, in aot_wrapper_synthetic_base
[rank3]:     return compiler_fn(flat_fn, flat_args, aot_config, fw_metadata=fw_metadata)
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py", line 123, in aot_dispatch_base
[rank3]:     fakified_out = _compute_output_meta_with_inductor_strides(
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py", line 70, in _compute_output_meta_with_inductor_strides
[rank3]:     out = [n.meta["val"] for n in (list(fw_module.graph.nodes)[-1].args[0])]
[rank3]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py", line 70, in <listcomp>
[rank3]:     out = [n.meta["val"] for n in (list(fw_module.graph.nodes)[-1].args[0])]
[rank3]: torch._dynamo.exc.BackendCompilerFailed: backend='compile_fn' raised:
[rank3]: AttributeError: 'int' object has no attribute 'meta'

[rank3]: While executing %submod_2 : [num_users=4] = call_module[target=submod_2](args = (%getitem, %s0, %s1, %getitem_1, %getitem_2, %getitem_3), kwargs = {})
[rank3]: Original traceback:
[rank3]: None

[rank3]: Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information


[rank3]: You can suppress this exception and fall back to eager by setting:
[rank3]:     import torch._dynamo
[rank3]:     torch._dynamo.config.suppress_errors = True

[rank0]: Traceback (most recent call last):
[rank0]:   File "/home/ubuntu/nanogpt-train/build_nanogpt/train_gpt2.py", line 444, in <module>
[rank0]:     logits, loss = model(tokens)
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/parallel/distributed.py", line 1593, in forward
[rank0]:     else self._run_ddp_forward(*inputs, **kwargs)
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/parallel/distributed.py", line 1411, in _run_ddp_forward
[rank0]:     return self.module(*inputs, **kwargs)  # type: ignore[index]
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py", line 451, in _fn
[rank0]:     return fn(*args, **kwargs)
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py", line 917, in catch_errors
[rank0]:     return hijacked_callback(frame, cache_entry, hooks, frame_state)
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py", line 786, in _convert_frame
[rank0]:     result = inner_convert(
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py", line 400, in _convert_frame_assert
[rank0]:     return _compile(
[rank0]:   File "/usr/lib/python3.10/contextlib.py", line 79, in inner
[rank0]:     return func(*args, **kwds)
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py", line 676, in _compile
[rank0]:     guarded_code = compile_inner(code, one_graph, hooks, transform)
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/utils.py", line 262, in time_wrapper
[rank0]:     r = func(*args, **kwargs)
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py", line 535, in compile_inner
[rank0]:     out_code = transform_code_object(code, transform)
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/bytecode_transformation.py", line 1036, in transform_code_object
[rank0]:     transformations(instructions, code_options)
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py", line 165, in _fn
[rank0]:     return fn(*args, **kwargs)
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py", line 500, in transform
[rank0]:     tracer.run()
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py", line 2149, in run
[rank0]:     super().run()
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py", line 810, in run
[rank0]:     and self.step()
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py", line 773, in step
[rank0]:     getattr(self, inst.opname)(inst)
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py", line 2268, in RETURN_VALUE
[rank0]:     self.output.compile_subgraph(
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/output_graph.py", line 1001, in compile_subgraph
[rank0]:     self.compile_and_call_fx_graph(tx, pass2.graph_output_vars(), root)
[rank0]:   File "/usr/lib/python3.10/contextlib.py", line 79, in inner
[rank0]:     return func(*args, **kwds)
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/output_graph.py", line 1178, in compile_and_call_fx_graph
[rank0]:     compiled_fn = self.call_user_compiler(gm)
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/utils.py", line 262, in time_wrapper
[rank0]:     r = func(*args, **kwargs)
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/output_graph.py", line 1251, in call_user_compiler
[rank0]:     raise BackendCompilerFailed(self.compiler_fn, e).with_traceback(
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/output_graph.py", line 1232, in call_user_compiler
[rank0]:     compiled_fn = compiler_fn(gm, self.example_inputs())
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/backends/distributed.py", line 606, in compile_fn
[rank0]:     submod_compiler.run(*example_inputs)
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/fx/interpreter.py", line 145, in run
[rank0]:     self.env[node] = self.run_node(node)
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/backends/distributed.py", line 348, in run_node
[rank0]:     compiled_submod_real = self.compile_submod(real_mod, new_args, kwargs)
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/backends/distributed.py", line 263, in compile_submod
[rank0]:     self.compiler(input_mod, args),
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/repro/after_dynamo.py", line 117, in debug_wrapper
[rank0]:     compiled_gm = compiler_fn(gm, example_inputs)
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/__init__.py", line 1731, in __call__
[rank0]:     return compile_fx(model_, inputs_, config_patches=self.config)
[rank0]:   File "/usr/lib/python3.10/contextlib.py", line 79, in inner
[rank0]:     return func(*args, **kwds)
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_inductor/compile_fx.py", line 1330, in compile_fx
[rank0]:     return aot_autograd(
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/backends/common.py", line 58, in compiler_fn
[rank0]:     cg = aot_module_simplified(gm, example_inputs, **kwargs)
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_functorch/aot_autograd.py", line 903, in aot_module_simplified
[rank0]:     compiled_fn = create_aot_dispatcher_function(
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/utils.py", line 262, in time_wrapper
[rank0]:     r = func(*args, **kwargs)
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_functorch/aot_autograd.py", line 628, in create_aot_dispatcher_function
[rank0]:     compiled_fn = compiler_fn(flat_fn, fake_flat_args, aot_config, fw_metadata=fw_metadata)
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py", line 443, in aot_wrapper_dedupe
[rank0]:     return compiler_fn(flat_fn, leaf_flat_args, aot_config, fw_metadata=fw_metadata)
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py", line 648, in aot_wrapper_synthetic_base
[rank0]:     return compiler_fn(flat_fn, flat_args, aot_config, fw_metadata=fw_metadata)
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py", line 123, in aot_dispatch_base
[rank0]:     fakified_out = _compute_output_meta_with_inductor_strides(
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py", line 70, in _compute_output_meta_with_inductor_strides
[rank0]:     out = [n.meta["val"] for n in (list(fw_module.graph.nodes)[-1].args[0])]
[rank0]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py", line 70, in <listcomp>
[rank0]:     out = [n.meta["val"] for n in (list(fw_module.graph.nodes)[-1].args[0])]
[rank0]: torch._dynamo.exc.BackendCompilerFailed: backend='compile_fn' raised:
[rank0]: AttributeError: 'int' object has no attribute 'meta'

[rank0]: While executing %submod_2 : [num_users=4] = call_module[target=submod_2](args = (%getitem, %s0, %s1, %getitem_1, %getitem_2, %getitem_3), kwargs = {})
[rank0]: Original traceback:
[rank0]: None

[rank0]: Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information


[rank0]: You can suppress this exception and fall back to eager by setting:
[rank0]:     import torch._dynamo
[rank0]:     torch._dynamo.config.suppress_errors = True

[rank4]: Traceback (most recent call last):
[rank4]:   File "/home/ubuntu/nanogpt-train/build_nanogpt/train_gpt2.py", line 444, in <module>
[rank4]:     logits, loss = model(tokens)
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
[rank4]:     return self._call_impl(*args, **kwargs)
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
[rank4]:     return forward_call(*args, **kwargs)
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/parallel/distributed.py", line 1593, in forward
[rank4]:     else self._run_ddp_forward(*inputs, **kwargs)
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/parallel/distributed.py", line 1411, in _run_ddp_forward
[rank4]:     return self.module(*inputs, **kwargs)  # type: ignore[index]
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
[rank4]:     return self._call_impl(*args, **kwargs)
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
[rank4]:     return forward_call(*args, **kwargs)
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py", line 451, in _fn
[rank4]:     return fn(*args, **kwargs)
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
[rank4]:     return self._call_impl(*args, **kwargs)
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
[rank4]:     return forward_call(*args, **kwargs)
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py", line 917, in catch_errors
[rank4]:     return hijacked_callback(frame, cache_entry, hooks, frame_state)
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py", line 786, in _convert_frame
[rank4]:     result = inner_convert(
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py", line 400, in _convert_frame_assert
[rank4]:     return _compile(
[rank4]:   File "/usr/lib/python3.10/contextlib.py", line 79, in inner
[rank4]:     return func(*args, **kwds)
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py", line 676, in _compile
[rank4]:     guarded_code = compile_inner(code, one_graph, hooks, transform)
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/utils.py", line 262, in time_wrapper
[rank4]:     r = func(*args, **kwargs)
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py", line 535, in compile_inner
[rank4]:     out_code = transform_code_object(code, transform)
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/bytecode_transformation.py", line 1036, in transform_code_object
[rank4]:     transformations(instructions, code_options)
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py", line 165, in _fn
[rank4]:     return fn(*args, **kwargs)
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py", line 500, in transform
[rank4]:     tracer.run()
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py", line 2149, in run
[rank4]:     super().run()
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py", line 810, in run
[rank4]:     and self.step()
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py", line 773, in step
[rank4]:     getattr(self, inst.opname)(inst)
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py", line 2268, in RETURN_VALUE
[rank4]:     self.output.compile_subgraph(
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/output_graph.py", line 1001, in compile_subgraph
[rank4]:     self.compile_and_call_fx_graph(tx, pass2.graph_output_vars(), root)
[rank4]:   File "/usr/lib/python3.10/contextlib.py", line 79, in inner
[rank4]:     return func(*args, **kwds)
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/output_graph.py", line 1178, in compile_and_call_fx_graph
[rank4]:     compiled_fn = self.call_user_compiler(gm)
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/utils.py", line 262, in time_wrapper
[rank4]:     r = func(*args, **kwargs)
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/output_graph.py", line 1251, in call_user_compiler
[rank4]:     raise BackendCompilerFailed(self.compiler_fn, e).with_traceback(
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/output_graph.py", line 1232, in call_user_compiler
[rank4]:     compiled_fn = compiler_fn(gm, self.example_inputs())
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/backends/distributed.py", line 606, in compile_fn
[rank4]:     submod_compiler.run(*example_inputs)
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/fx/interpreter.py", line 145, in run
[rank4]:     self.env[node] = self.run_node(node)
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/backends/distributed.py", line 348, in run_node
[rank4]:     compiled_submod_real = self.compile_submod(real_mod, new_args, kwargs)
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/backends/distributed.py", line 263, in compile_submod
[rank4]:     self.compiler(input_mod, args),
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/repro/after_dynamo.py", line 117, in debug_wrapper
[rank4]:     compiled_gm = compiler_fn(gm, example_inputs)
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/__init__.py", line 1731, in __call__
[rank4]:     return compile_fx(model_, inputs_, config_patches=self.config)
[rank4]:   File "/usr/lib/python3.10/contextlib.py", line 79, in inner
[rank4]:     return func(*args, **kwds)
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_inductor/compile_fx.py", line 1330, in compile_fx
[rank4]:     return aot_autograd(
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/backends/common.py", line 58, in compiler_fn
[rank4]:     cg = aot_module_simplified(gm, example_inputs, **kwargs)
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_functorch/aot_autograd.py", line 903, in aot_module_simplified
[rank4]:     compiled_fn = create_aot_dispatcher_function(
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_dynamo/utils.py", line 262, in time_wrapper
[rank4]:     r = func(*args, **kwargs)
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_functorch/aot_autograd.py", line 628, in create_aot_dispatcher_function
[rank4]:     compiled_fn = compiler_fn(flat_fn, fake_flat_args, aot_config, fw_metadata=fw_metadata)
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py", line 443, in aot_wrapper_dedupe
[rank4]:     return compiler_fn(flat_fn, leaf_flat_args, aot_config, fw_metadata=fw_metadata)
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py", line 648, in aot_wrapper_synthetic_base
[rank4]:     return compiler_fn(flat_fn, flat_args, aot_config, fw_metadata=fw_metadata)
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py", line 123, in aot_dispatch_base
[rank4]:     fakified_out = _compute_output_meta_with_inductor_strides(
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py", line 70, in _compute_output_meta_with_inductor_strides
[rank4]:     out = [n.meta["val"] for n in (list(fw_module.graph.nodes)[-1].args[0])]
[rank4]:   File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py", line 70, in <listcomp>
[rank4]:     out = [n.meta["val"] for n in (list(fw_module.graph.nodes)[-1].args[0])]
[rank4]: torch._dynamo.exc.BackendCompilerFailed: backend='compile_fn' raised:
[rank4]: AttributeError: 'int' object has no attribute 'meta'

[rank4]: While executing %submod_2 : [num_users=4] = call_module[target=submod_2](args = (%getitem, %s0, %s1, %getitem_1, %getitem_2, %getitem_3), kwargs = {})
[rank4]: Original traceback:
[rank4]: None

[rank4]: Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information


[rank4]: You can suppress this exception and fall back to eager by setting:
[rank4]:     import torch._dynamo
[rank4]:     torch._dynamo.config.suppress_errors = True

W0616 20:34:34.710000 140609446490112 torch/distributed/elastic/multiprocessing/api.py:851] Sending process 76627 closing signal SIGTERM
W0616 20:34:34.710000 140609446490112 torch/distributed/elastic/multiprocessing/api.py:851] Sending process 76629 closing signal SIGTERM
W0616 20:34:34.710000 140609446490112 torch/distributed/elastic/multiprocessing/api.py:851] Sending process 76630 closing signal SIGTERM
W0616 20:34:34.710000 140609446490112 torch/distributed/elastic/multiprocessing/api.py:851] Sending process 76631 closing signal SIGTERM
W0616 20:34:34.710000 140609446490112 torch/distributed/elastic/multiprocessing/api.py:851] Sending process 76632 closing signal SIGTERM
W0616 20:34:34.710000 140609446490112 torch/distributed/elastic/multiprocessing/api.py:851] Sending process 76633 closing signal SIGTERM
E0616 20:34:38.106000 140609446490112 torch/distributed/elastic/multiprocessing/api.py:826] failed (exitcode: 1) local_rank: 0 (pid: 76626) of binary: /usr/bin/python3
Traceback (most recent call last):
  File "/home/ubuntu/.local/bin/torchrun", line 8, in <module>
    sys.exit(main())
  File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 347, in wrapper
    return f(*args, **kwargs)
  File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/distributed/run.py", line 879, in main
    run(args)
  File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/distributed/run.py", line 870, in run
    elastic_launch(
  File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 132, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 263, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
train_gpt2.py FAILED
------------------------------------------------------------
Failures:
[1]:
  time      : 2024-06-16_20:34:34
  host      : 129-153-214-211
  rank      : 2 (local_rank: 2)
  exitcode  : 1 (pid: 76628)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2024-06-16_20:34:34
  host      : 129-153-214-211
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 76626)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
```