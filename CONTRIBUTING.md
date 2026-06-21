# Contributing to shwfs-ao-pipeline

Thank you for your interest in contributing!

## How to contribute

1. Fork the repository
2. Create a feature branch: `git checkout -b fix/your-fix-name`
3. Make changes, add tests if applicable
4. Ensure all tests pass: `pytest tests/`
5. Open a pull request against `main`

## Branch naming

- `fix/` — bug fixes
- `feat/` — new features
- `docs/` — documentation
- `ci/` — CI/workflow changes
- `chore/` — maintenance

## Commit style

Use conventional commits:
```
fix(module): short description
feat(module): short description
docs: short description
```

## Running tests

```bash
pip install -r requirements.txt
pytest tests/
```

## Reporting bugs

Open an issue with:
- What you expected
- What actually happened
- Config and Python version
