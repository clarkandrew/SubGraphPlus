# 🤝 Contributing to SubgraphRAG+

Thank you for your interest in contributing to SubgraphRAG+! This guide will help you get started with development and understand our contribution process.

## 🚀 Quick Start for Contributors

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/your-username/SubgraphRAGPlus.git
cd SubgraphRAGPlus

# Add the original repository as upstream
git remote add upstream https://github.com/original-owner/SubgraphRAGPlus.git
```

### 2. Development Setup

```bash
# Setup development environment
make setup-dev

# Or manual setup:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .

# Install pre-commit hooks
pre-commit install
```

### 3. Verify Setup

```bash
# Run tests to ensure everything works
make test

# Check code quality
make lint

# Start the development server
make serve
```

## 📋 Development Workflow

### Creating a Feature Branch

```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create a feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes:
git checkout -b fix/issue-description
```

### Making Changes

1. **Write tests first** (TDD approach recommended)
2. **Implement your changes**
3. **Run tests and quality checks**
4. **Update documentation** if needed
5. **Commit with clear messages**

```bash
# Run quality checks before committing
make quality test

# Commit your changes
git add .
git commit -m "feat: add new feature description"

# Push to your fork
git push origin feature/your-feature-name
```

### Submitting a Pull Request

1. **Push your branch** to your fork
2. **Create a Pull Request** on GitHub
3. **Fill out the PR template** completely
4. **Respond to review feedback** promptly
5. **Keep your branch updated** with main

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests for component interactions
├── e2e/           # End-to-end tests for complete workflows
└── fixtures/      # Test data and fixtures
```

### Writing Tests

```python
# Example unit test
import pytest
from app.retriever import HybridRetriever

class TestHybridRetriever:
    def test_retrieval_with_valid_query(self):
        retriever = HybridRetriever()
        results = retriever.retrieve("test query")
        assert len(results) > 0
        assert all(hasattr(r, 'score') for r in results)

    @pytest.mark.asyncio
    async def test_async_retrieval(self):
        retriever = HybridRetriever()
        results = await retriever.retrieve_async("test query")
        assert isinstance(results, list)
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test categories
make test-integration
pytest tests/unit/ -v

# Run with coverage
make test-coverage

# Run tests for specific file
pytest tests/unit/test_retriever.py -v
```

## 📝 Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 88 characters (Black default)
- **Import order**: isort configuration in `pyproject.toml`
- **Type hints**: Required for all public functions
- **Docstrings**: Google style for all public classes and functions

### Code Formatting

```bash
# Auto-format code
make format

# Check formatting
black --check src/ tests/ scripts/
isort --check src/ tests/ scripts/
```

### Type Checking

```bash
# Run type checking
make typecheck

# Or directly:
mypy src/ --ignore-missing-imports
```

### Example Code Style

```python
"""Module for hybrid retrieval functionality."""

from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combines graph traversal and semantic search for optimal retrieval.
    
    This class implements a hybrid approach that leverages both structured
    graph relationships and dense vector similarity for comprehensive
    knowledge retrieval.
    
    Args:
        graph_weight: Weight for graph-based results (0.0-1.0)
        semantic_weight: Weight for semantic search results (0.0-1.0)
        max_results: Maximum number of results to return
    """
    
    def __init__(
        self,
        graph_weight: float = 0.6,
        semantic_weight: float = 0.4,
        max_results: int = 10
    ) -> None:
        self.graph_weight = graph_weight
        self.semantic_weight = semantic_weight
        self.max_results = max_results
        
    def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant information using hybrid approach.
        
        Args:
            query: The search query string
            filters: Optional filters to apply to results
            
        Returns:
            List of retrieved results with relevance scores
            
        Raises:
            ValueError: If query is empty or invalid
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")
            
        logger.info(f"Retrieving results for query: {query[:50]}...")
        
        # Implementation here
        return []
```

## 🏗️ Architecture Guidelines

### Project Structure

```
src/app/
├── api.py          # FastAPI routes and endpoints
├── models.py       # Pydantic data models
├── config.py       # Configuration management
├── database.py     # Database connections
├── retriever.py    # Core retrieval logic
└── ml/            # Machine learning components
    ├── embeddings.py
    ├── models.py
    └── utils.py
```

### Design Principles

1. **Separation of Concerns**: Each module has a single responsibility
2. **Dependency Injection**: Use dependency injection for testability
3. **Configuration Management**: All settings in config files
4. **Error Handling**: Comprehensive error handling with proper logging
5. **Performance**: Async/await for I/O operations, caching where appropriate

### Adding New Features

1. **API Endpoints**: Add to `app/api.py` with proper validation
2. **Data Models**: Define in `app/models.py` using Pydantic
3. **Business Logic**: Implement in appropriate modules
4. **Database Changes**: Create migration scripts
5. **Tests**: Add comprehensive test coverage

## 📚 Documentation Standards

### Code Documentation

- **Docstrings**: Required for all public classes and functions
- **Type hints**: Required for all function parameters and returns
- **Comments**: Explain complex logic, not obvious code
- **README updates**: Update if adding new features or changing setup

### Documentation Files

- **API changes**: Update `docs/api_reference.md`
- **Architecture changes**: Update `docs/architecture.md`
- **New features**: Add usage examples to `docs/api_examples.md`
- **Breaking changes**: Document in migration guide

## 🔍 Code Review Process

### Before Submitting

- [ ] All tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] No sensitive information in code
- [ ] Performance impact considered

### Review Criteria

1. **Functionality**: Does the code work as intended?
2. **Testing**: Are there adequate tests?
3. **Performance**: Any performance implications?
4. **Security**: Are there security considerations?
5. **Maintainability**: Is the code readable and maintainable?

### Responding to Reviews

- **Be responsive**: Address feedback promptly
- **Ask questions**: If feedback is unclear, ask for clarification
- **Make changes**: Update code based on feedback
- **Test changes**: Ensure fixes don't break anything
- **Update PR**: Push changes and notify reviewers

## 🐛 Bug Reports

### Before Reporting

1. **Search existing issues** for similar problems
2. **Try the latest version** to see if it's already fixed
3. **Check documentation** for known issues
4. **Gather debug information** using our diagnostic tools

### Bug Report Template

```markdown
**Bug Description**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. See error

**Expected Behavior**
What you expected to happen.

**Environment**
- OS: [e.g. macOS 12.0]
- Python version: [e.g. 3.11.0]
- Docker version: [e.g. 20.10.17]
- SubgraphRAG+ version: [e.g. 1.0.0]

**Additional Context**
- Error logs
- Screenshots
- Configuration files (remove sensitive data)
```

## 💡 Feature Requests

### Before Requesting

1. **Check existing issues** for similar requests
2. **Consider the scope** - is this a core feature?
3. **Think about implementation** - how would this work?
4. **Consider alternatives** - are there existing solutions?

### Feature Request Template

```markdown
**Feature Description**
A clear description of the feature you'd like to see.

**Use Case**
Describe the problem this feature would solve.

**Proposed Solution**
How you envision this feature working.

**Alternatives Considered**
Other solutions you've considered.

**Additional Context**
Any other context about the feature request.
```

## 🏷️ Commit Message Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples

```bash
feat(api): add streaming response support
fix(database): resolve Neo4j connection timeout
docs(readme): update installation instructions
test(retriever): add unit tests for hybrid search
```

## 🚀 Release Process

### Version Numbering

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in appropriate files
- [ ] Release notes prepared
- [ ] Docker images built and tested

## 🆘 Getting Help

### Community Support

- **💬 GitHub Discussions**: General questions and community help
- **🐛 GitHub Issues**: Bug reports and feature requests
- **📖 Documentation**: Comprehensive guides in `docs/`

### Development Questions

- **Architecture decisions**: Check `docs/architecture.md`
- **API usage**: See `docs/api_reference.md`
- **Setup issues**: Refer to `docs/troubleshooting.md`

### Contact

- **Maintainers**: Tag `@maintainers` in issues
- **Security issues**: Email security@subgraphrag.com
- **General questions**: Use GitHub Discussions

## 📜 Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

### Our Standards

- **Be respectful** and inclusive
- **Be collaborative** and constructive
- **Be patient** with newcomers
- **Be professional** in all interactions

---

**🎉 Thank you for contributing to SubgraphRAG+! Your contributions help make this project better for everyone.** 