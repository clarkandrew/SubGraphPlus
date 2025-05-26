# 🤝 Contributing to SubgraphRAG+

Thank you for your interest in contributing to SubgraphRAG+! This guide will help you get started with contributing to our knowledge graph-powered question answering system.

## 🌟 Ways to Contribute

- **🐛 Bug Reports**: Help us identify and fix issues
- **💡 Feature Requests**: Suggest new capabilities and improvements
- **📝 Documentation**: Improve guides, examples, and API documentation
- **🧪 Testing**: Add test cases and improve test coverage
- **💻 Code**: Implement new features, fix bugs, optimize performance
- **🎨 UI/UX**: Enhance the frontend interface and user experience
- **📊 Examples**: Create tutorials, demos, and use case examples

## 🚀 Quick Start for Contributors

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/your-username/SubgraphRAGPlus.git
cd SubgraphRAGPlus

# Add upstream remote
git remote add upstream https://github.com/original-owner/SubgraphRAGPlus.git
```

### 2. Set Up Development Environment

```bash
# Use our automated setup script
./bin/setup_dev.sh --run-tests

# Or manually:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Verify Your Setup

```bash
# Run tests to ensure everything works
make test

# Run the demo to verify functionality
python examples/demo_quickstart.py --skip-neo4j --skip-data

# Check code style
make lint
```

### 4. Create a Feature Branch

```bash
# Create and switch to a new branch
git checkout -b feature/your-feature-name

# Or for bug fixes:
git checkout -b fix/issue-description
```

## 📋 Development Guidelines

### Code Style

We follow Python best practices and use automated tools:

```bash
# Format code with Black
make format

# Check style with flake8 and mypy
make lint

# Sort imports with isort
isort src/ tests/

# All checks together
make check-all
```

**Style Requirements:**
- **Black** for code formatting (line length: 88)
- **flake8** for linting
- **mypy** for type checking
- **isort** for import sorting
- **Conventional Commits** for commit messages

### Testing Requirements

All contributions must include appropriate tests:

```bash
# Run all tests
make test

# Run specific test categories
python -m pytest tests/test_api.py -v          # API tests
python -m pytest tests/test_retriever.py -v    # Core logic tests
python -m pytest tests/test_integration.py -v  # Integration tests

# Run with coverage
make test-coverage
```

**Testing Standards:**
- **Unit tests** for individual functions/classes
- **Integration tests** for component interactions
- **API tests** for endpoint functionality
- **Minimum 80% code coverage** for new code
- **Test naming**: `test_<function>_<scenario>_<expected_result>`

### Documentation Requirements

- **Docstrings**: All public functions must have comprehensive docstrings
- **Type hints**: Use type annotations for all function parameters and returns
- **README updates**: Update relevant documentation for user-facing changes
- **API docs**: Update OpenAPI specs for API changes

Example docstring format:
```python
def extract_entities(text: str, model_name: str = "default") -> List[Entity]:
    """Extract named entities from text using the specified model.
    
    Args:
        text: Input text to process
        model_name: Name of the NER model to use
        
    Returns:
        List of extracted entities with types and confidence scores
        
    Raises:
        ModelNotFoundError: If the specified model is not available
        
    Example:
        >>> entities = extract_entities("Apple Inc. was founded by Steve Jobs")
        >>> print(entities[0].text)
        "Apple Inc."
    """
```

## 🏗️ Architecture Overview

Understanding the codebase structure helps with contributions:

```
SubgraphRAGPlus/
├── src/app/                 # Core application code
│   ├── api.py              # FastAPI REST endpoints
│   ├── retriever.py        # Hybrid retrieval engine
│   ├── database.py         # Neo4j & SQLite connections
│   ├── ml/                 # ML models (LLM, embeddings, MLP)
│   └── utils.py            # Shared utilities
├── scripts/                # Utility scripts and tools
├── tests/                  # Comprehensive test suite
├── docs/                   # Documentation
├── frontend/               # Next.js frontend (optional)
├── config/                 # Configuration files
└── examples/               # Demo scripts and examples
```

### Key Components

- **API Layer** (`src/app/api.py`): REST endpoints and request handling
- **Retrieval Engine** (`src/app/retriever.py`): Core RAG logic
- **Database Layer** (`src/app/database.py`): Data persistence and queries
- **ML Models** (`src/app/ml/`): LLM, embedding, and MLP model interfaces
- **Configuration** (`src/app/config.py`): Centralized settings management

## 🐛 Bug Reports

### Before Submitting

1. **Search existing issues** to avoid duplicates
2. **Try the latest version** to see if the issue is already fixed
3. **Check documentation** to ensure it's not expected behavior
4. **Run diagnostics** to gather relevant information

### Bug Report Template

```markdown
**Bug Description**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Run command '...'
2. Send request '...'
3. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g., macOS 12.0, Ubuntu 20.04]
- Python version: [e.g., 3.11.0]
- SubgraphRAG+ version: [e.g., v0.9.0]
- Installation method: [Docker/Development/Demo]

**Logs**
```
Paste relevant log output here
```

**Additional Context**
Any other context about the problem.
```

## 💡 Feature Requests

### Before Submitting

1. **Check existing issues** and discussions
2. **Review the roadmap** to see if it's already planned
3. **Consider the scope** - is this a core feature or plugin?
4. **Think about backwards compatibility**

### Feature Request Template

```markdown
**Feature Description**
A clear description of the feature you'd like to see.

**Use Case**
Describe the problem this feature would solve.

**Proposed Solution**
How you envision this feature working.

**Alternatives Considered**
Other approaches you've considered.

**Implementation Notes**
Technical considerations or suggestions.
```

## 💻 Code Contributions

### Development Workflow

1. **Create an issue** (for significant changes)
2. **Fork and clone** the repository
3. **Create a feature branch** from `main`
4. **Make your changes** following our guidelines
5. **Add tests** for new functionality
6. **Update documentation** as needed
7. **Run the full test suite**
8. **Submit a pull request**

### Pull Request Process

#### Before Submitting

```bash
# Ensure your branch is up to date
git fetch upstream
git rebase upstream/main

# Run all checks
make check-all
make test

# Test the demo still works
python examples/demo_quickstart.py --skip-neo4j --skip-data
```

#### PR Template

```markdown
**Description**
Brief description of changes.

**Type of Change**
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

**Testing**
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Demo script works
- [ ] Manual testing completed

**Checklist**
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

### Commit Message Format

We use [Conventional Commits](https://conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(api): add streaming response support for /query endpoint

fix(retriever): resolve token budget enforcement in greedy_connect_v2

docs(readme): update installation instructions for demo script

test(integration): add comprehensive API endpoint tests
```

## 🧪 Testing Guidelines

### Test Categories

1. **Unit Tests** (`tests/unit/`): Test individual functions and classes
2. **Integration Tests** (`tests/integration/`): Test component interactions
3. **API Tests** (`tests/api/`): Test REST endpoints
4. **End-to-End Tests** (`tests/e2e/`): Test complete workflows

### Writing Tests

```python
import pytest
from unittest.mock import Mock, patch
from src.app.retriever import HybridRetriever

class TestHybridRetriever:
    """Test suite for HybridRetriever class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        config = Mock()
        config.retrieval_token_budget = 1000
        config.max_dde_hops = 2
        return config
    
    def test_retrieve_entities_success(self, mock_config):
        """Test successful entity retrieval."""
        # Arrange
        retriever = HybridRetriever(mock_config)
        query = "What is machine learning?"
        
        # Act
        result = retriever.retrieve_entities(query)
        
        # Assert
        assert len(result) > 0
        assert all(entity.confidence > 0 for entity in result)
    
    def test_retrieve_entities_empty_query(self, mock_config):
        """Test retrieval with empty query."""
        # Arrange
        retriever = HybridRetriever(mock_config)
        
        # Act & Assert
        with pytest.raises(ValueError, match="Query cannot be empty"):
            retriever.retrieve_entities("")
```

### Test Configuration

Use the `TESTING=1` environment variable for faster tests:

```bash
# Skip expensive operations during testing
TESTING=1 python -m pytest tests/
```

## 📚 Documentation Contributions

### Documentation Structure

- **README.md**: Project overview and quick start
- **docs/**: Detailed documentation
  - `installation.md`: Setup instructions
  - `architecture.md`: System design
  - `api_reference.md`: API documentation
  - `troubleshooting.md`: Common issues and solutions
- **Code comments**: Inline documentation
- **Docstrings**: Function/class documentation

### Documentation Standards

- **Clear and concise**: Use simple language
- **Examples included**: Provide code examples
- **Up to date**: Keep in sync with code changes
- **Accessible**: Consider different skill levels
- **Searchable**: Use clear headings and structure

## 🎨 Frontend Contributions

The frontend is built with Next.js and shadcn/ui:

```bash
# Setup frontend development
cd frontend
npm install
npm run dev

# Run frontend tests
npm test

# Build for production
npm run build
```

### Frontend Guidelines

- **TypeScript**: Use TypeScript for all new code
- **Component structure**: Follow shadcn/ui patterns
- **Responsive design**: Ensure mobile compatibility
- **Accessibility**: Follow WCAG guidelines
- **Performance**: Optimize for speed and bundle size

## 🏷️ Release Process

### Versioning

We use [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

### Release Checklist

1. **Update version** in relevant files
2. **Update CHANGELOG.md** with release notes
3. **Run full test suite** on multiple platforms
4. **Update documentation** as needed
5. **Create release tag** and GitHub release
6. **Announce release** in discussions

## 🤔 Getting Help

### Community Support

- **GitHub Discussions**: General questions and ideas
- **GitHub Issues**: Bug reports and feature requests
- **Documentation**: Check existing guides first

### Maintainer Contact

For sensitive issues or maintainer-specific questions:
- Create a private issue with the `maintainer` label
- Email: [maintainer-email@example.com]

## 📜 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of background, experience level, or identity.

### Expected Behavior

- **Be respectful** and considerate in all interactions
- **Be collaborative** and help others learn
- **Be constructive** when providing feedback
- **Be patient** with newcomers and questions
- **Be professional** in all communications

### Unacceptable Behavior

- Harassment, discrimination, or offensive language
- Personal attacks or trolling
- Spam or off-topic content
- Sharing private information without permission

### Enforcement

Violations will be addressed by maintainers and may result in temporary or permanent bans from the project.

## 🙏 Recognition

### Contributors

All contributors are recognized in:
- **README.md**: Contributors section
- **CHANGELOG.md**: Release notes
- **GitHub**: Contributor graphs and statistics

### Types of Contributions

We value all types of contributions:
- **Code**: Features, fixes, optimizations
- **Documentation**: Guides, examples, improvements
- **Testing**: Test cases, bug reports, validation
- **Community**: Helping others, discussions, feedback
- **Design**: UI/UX improvements, graphics, branding

---

## 🚀 Ready to Contribute?

1. **Star the repository** to show your support
2. **Read the documentation** to understand the project
3. **Set up your development environment**
4. **Look for "good first issue" labels** for beginner-friendly tasks
5. **Join the discussions** to connect with the community

Thank you for contributing to SubgraphRAG+! 🎉 