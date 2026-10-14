"""End-to-end pipeline orchestrator.

Chains the agentic-scientist workflow phases:

1. task drafting (``src/scripts/task.sh``)
2. investigation (``src.investigation.cli``, ``agsci investigate``)
3. ablations and baselines (``python -m src.variants_pipeline.cli``)
4. final evaluation of the selected program
5. best-effort post-hoc analysis (``python -m src.posthoc_pipeline.cli``)
6. paper writing (``src/scripts/paper.sh``)
7. optional final AI review (shared managed reviewer)

into a single Hydra-configured command with named phase selection and resume.
See :mod:`src.pipeline.cli` for the entry point.
"""
