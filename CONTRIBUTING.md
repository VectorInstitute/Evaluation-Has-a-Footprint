# Contributing

Thanks for your interest in contributing.

To submit PRs, please fill out the PR template along with the PR. If the PR
fixes an issue, don't forget to link the PR to the issue!

## Pre-commit hooks

Once the python virtual environment is setup, you can run pre-commit hooks using:

```bash
pre-commit run --all-files
```

## Coding guidelines

For code style, we recommend PEP 8.

For docstrings we use numpy format.

We use ruff for code formatting and static code analysis. The pre-commit hooks
show errors which you need to fix before submitting a PR.

We use type hints in our code which is then checked using mypy.
