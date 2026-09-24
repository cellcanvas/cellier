"""GPU cache budgets for tests that build multiscale visuals.

A multiscale visual allocates its whole ``gpu_budget_bytes`` up front (1 GiB
for the 3D brick cache by default), whatever the size of its data.  On Linux
CI the GPU is lavapipe, so that is host RAM, and a closed viewer does not hand
it back: with default budgets the test session ran the 16 GB runner out of
memory.  Pass ``**SMALL_BUDGETS`` to ``MultiscaleImageRenderConfig`` or
``MultiscaleLabelRenderConfig``.

64 MiB holds 343 bricks at ``block_size=32`` and 15625 at ``block_size=8``,
more than any test dataset needs.
"""

BUDGET_3D = 64 * 1024**2
BUDGET_2D = 16 * 1024**2
SMALL_BUDGETS = {"gpu_budget_bytes": BUDGET_3D, "gpu_budget_bytes_2d": BUDGET_2D}
