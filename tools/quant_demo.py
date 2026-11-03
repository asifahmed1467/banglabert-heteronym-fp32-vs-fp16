"""Quantization basics in 5 small experiments. Needs only numpy:  py quant_demo.py"""
import numpy as np

np.set_printoptions(precision=8, suppress=False)

print("1) SIZE: bytes per number")
w = np.random.default_rng(0).normal(0, 0.05, 1_000_000).astype(np.float32)   # 1 million fake 'weights'
print(f"   fp32: {w.nbytes/1e6:.1f} MB   fp16: {w.astype(np.float16).nbytes/1e6:.1f} MB   int8: {w.size/1e6:.1f} MB\n")

print("2) ROUNDING: fp32 -> fp16 loses digits (this is the only change the model sees)")
x = np.array([0.123456789, 0.000123456789, 3.14159265], dtype=np.float32)
print("   fp32:", x)
print("   fp16:", x.astype(np.float16).astype(np.float32), "\n")

h = w.astype(np.float16).astype(np.float32)
err = np.abs(w - h)
print(f"3) ERROR over {w.size:,} weights: mean {err.mean():.2e}, max {err.max():.2e}, "
      f"relative L2 {100*np.linalg.norm(w-h)/np.linalg.norm(w):.4f}%   <- same kind of number the benchmark page shows\n")

print("4) LIMITS of fp16: biggest value 65504, tiny values can become 0")
print("   70000 as fp16 ->", np.float32(70000).astype(np.float16), " (overflow)")
print("   1e-9  as fp16 ->", np.float32(1e-9).astype(np.float16), " (rounded to zero)\n")

print("5) int8 (for comparison): 1 byte, scale = max|w| / 127")
scale = np.abs(w).max() / 127
q = np.round(w / scale).astype(np.int8)
back = q.astype(np.float32) * scale
print(f"   relative L2 error {100*np.linalg.norm(w-back)/np.linalg.norm(w):.2f}%  (much bigger than fp16's, so int8 needs accuracy re-checks)")
