"""Cross-layer integration tests for the transform and slicing path.

This directory is deliberately cross-layer: its suites drive a model, a
data store and a render-layer visual together, so none of the layer-scoped
test directories (``render``, ``gui``, ``visuals``, ``v2``) fits.  Same
reasoning that put ``tests/transform/`` at the top level.

The golden baseline recorded here (``baseline/*.json`` +
``test_golden_baseline.py``) is a replayable snapshot of what every
drivable render-layer visual family produces -- a node matrix and a
tuple of datastore selections -- for a parameterised matrix of
transforms and dims states.  Changes to the slicing path assert against
it, which is what turns "did I break the renderer" from a judgement call
into an exit code.
"""
