# Contributing

Please open an issue describing the input, expected output, model ID and software versions. Do not include credentials, private images or server addresses.

Use a project-local virtual environment. Run `pytest -q` for protocol changes. Model-adapter changes also require a real-GPU first-step parity report from `scripts/validate_model.py`; keep incorrect probe answers in the evidence. Report interface verification separately from benchmark accuracy, training and flight outcomes.

Model weights, caches, runtime logs and local secrets must stay outside commits. Submit focused pull requests with the behavior change and relevant verification.
