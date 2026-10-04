# Contributing

Thank you for your interest in contributing! This project is primarily a personal learning portfolio, but thoughtful contributions are welcome.

## How to Contribute

### Bug Reports
- Use the issue template
- Include: OS, Python/Node version, steps to reproduce, expected vs actual behavior
- For security issues, see [SECURITY.md](SECURITY.md) instead

### Feature Requests
- Open an issue with the "enhancement" label
- Describe the problem you're solving, not just the solution
- Explain why it fits the project's scope

### Pull Requests
1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Make focused, atomic commits with clear messages
4. Run tests: `pytest` / `npm test` / `cargo test`
5. Ensure CI passes
6. Open PR with description of changes and rationale

### Code Style
- **Python**: Black (88 chars), type hints, mypy clean
- **JavaScript/TypeScript**: ESLint + Prettier, strict TS
- **Go**: gofmt, go vet, golint clean
- **Rust**: rustfmt, clippy clean

### Testing
- New features require tests
- Bug fixes require regression tests
- Aim for >80% coverage on new code
- Run full test suite before PR

### Documentation
- Update README for user-facing changes
- Add docstrings/comments for public APIs
- Update CHANGELOG.md if applicable

## License

By contributing, you agree your contributions will be licensed under the project's license (MIT unless otherwise noted).
