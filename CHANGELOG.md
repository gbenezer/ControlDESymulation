# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Dependabot configuration for automated dependency updates
- VS Code workspace settings for development environment
- Pre-commit hooks configuration

### Changed

### Deprecated

### Removed

### Fixed

### Security

## [1.0.0] - 2025-01-07

### Added
- Initial release of ControlDESymulation (cdesym)
- Symbolic dynamical systems framework with multi-backend numerical execution
- Core system classes: `DynamicalSystem`, `StateSpace`, `ControlSystem`
- Multiple integration backends:
  - NumPy/SciPy backend for basic numerical integration
  - PyTorch backend with `torchdiffeq` and `torchsde` for GPU-accelerated ODE/SDE solving
  - JAX backend with `diffrax` for JIT-compiled integration
  - Julia backend with `DifferentialEquations.jl` via `diffeqpy`
- Integration methods: Euler, Midpoint, Heun, RK4, and adaptive solvers
- Stochastic differential equation (SDE) support with multiple noise types
- Control theory utilities:
  - LQR controller design
  - Lyapunov stability analysis
  - Certificate function verification
- Visualization framework with Plotly and Matplotlib support
- Type system architecture with Pydantic validation
- Comprehensive test suite (unit and integration tests)
- Quarto-based documentation with tutorials and API reference
- CI/CD pipeline with GitHub Actions:
  - Multi-Python version testing (3.9-3.12)
  - Code quality checks (Ruff, Black, mypy)
  - Automatic documentation deployment to GitHub Pages
  - PyPI publishing on release

### Technical Details
- Source layout: `src/cdesym/`
- Minimum Python version: 3.9
- Core dependencies: NumPy 2.0+, SymPy 1.12+, SciPy 1.10+, Pydantic 2.5+

[Unreleased]: https://github.com/gbenezer/ControlDESymulation/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/gbenezer/ControlDESymulation/releases/tag/v1.0.0
