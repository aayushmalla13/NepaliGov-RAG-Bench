# Contributing Guide

Thank you for your interest in contributing to the NepaliGov-RAG-Bench project! This guide will help you get started with contributing to our open-source project.

## 🤝 How to Contribute

### Getting Started

1. **Fork the Repository**: Click the "Fork" button on the GitHub repository page
2. **Clone Your Fork**: 
   ```bash
   git clone https://github.com/YOUR_USERNAME/NepaliGov-RAG-Bench.git
   cd NepaliGov-RAG-Bench
   ```
3. **Set Up Development Environment**: Follow the installation guide in the documentation
4. **Create a Branch**: 
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Development Setup

#### Using Docker (Recommended)
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/NepaliGov-RAG-Bench.git
cd NepaliGov-RAG-Bench

# Build and run with Docker
./docker-build.sh
./docker-run.sh

# Access the application
open http://localhost:8093
```

#### Manual Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Run the application
python enhanced_web_app.py
```

## 📝 Contribution Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and modular

### Commit Messages

Use clear, descriptive commit messages:
```bash
# Good examples:
git commit -m "Add health check endpoint for Docker monitoring"
git commit -m "Fix OCR processing for Nepali language documents"
git commit -m "Update documentation with Docker setup instructions"

# Avoid:
git commit -m "fix"
git commit -m "updates"
```

### Pull Request Process

1. **Create a Pull Request**: From your forked repository to the main repository
2. **Provide Description**: Clearly describe what changes you made and why
3. **Link Issues**: Reference any related issues using `#issue_number`
4. **Test Your Changes**: Ensure your changes work correctly
5. **Update Documentation**: Update relevant documentation if needed

### Testing

Before submitting a pull request:
- Test your changes thoroughly
- Ensure all existing functionality still works
- Add tests for new features when possible
- Check that the application starts without errors

## 🐛 Reporting Issues

When reporting issues, please include:

1. **Clear Description**: What problem are you experiencing?
2. **Steps to Reproduce**: How can we reproduce the issue?
3. **Expected Behavior**: What should happen?
4. **Actual Behavior**: What actually happens?
5. **Environment Details**: OS, Python version, Docker version (if applicable)
6. **Error Messages**: Any error messages or logs

### Issue Templates

Use the provided issue templates when creating new issues:
- Bug Report
- Feature Request
- Documentation Improvement
- Question

## 🚀 Development Workflow

### Phase-Based Development

This project follows a 12-phase development approach:

1. **Phase 1**: Authority Detection
2. **Phase 2**: Document Types & Filtering
3. **Phase 3**: OCR Processing
4. **Phase 4**: Vector Storage
5. **Phase 5**: Semantic Search
6. **Phase 6**: Language Preference
7. **Phase 7**: QA Citation
8. **Phase 8**: Metrics & Evaluation
9. **Phase 9**: Language Selection
10. **Phase 10**: Refusal Handling
11. **Phase 11**: API & UI
12. **Phase 12**: Batch Processing

### Working on Phases

When contributing to a specific phase:
- Understand the phase's purpose and requirements
- Follow the existing patterns and architecture
- Ensure backward compatibility
- Update relevant documentation

## 👥 Contributors

We would like to thank our contributors who have made significant contributions to this project:

- **aayushmalla13** - [@aayushmalla13](https://github.com/aayushmalla13)
- **shijalsharmapoudel** - [@shijalsharmapoudel](https://github.com/shijalsharmapoudel)
- **babin411** - [@babin411](https://github.com/babin411)

### Contributing Areas

We welcome contributions in the following areas:

- **Documentation**: Improving guides, tutorials, and API documentation
- **Testing**: Adding unit tests, integration tests, and test coverage
- **Bug Fixes**: Identifying and fixing issues
- **Feature Development**: Adding new functionality following the phase architecture
- **Performance Optimization**: Improving system performance and efficiency
- **UI/UX Improvements**: Enhancing user experience and interface design
- **Docker & Deployment**: Improving containerization and deployment processes
- **Translation**: Adding support for additional languages

## 🛠️ Development Tools

### Recommended Tools

- **IDE**: VS Code, PyCharm, or any Python-compatible editor
- **Version Control**: Git with GitHub
- **Containerization**: Docker and Docker Compose
- **Documentation**: MkDocs with Material theme
- **Testing**: pytest for Python testing

### Useful Commands

```bash
# Check code style
flake8 src/
black src/

# Run tests
pytest tests/

# Build documentation
mkdocs serve

# Docker development
docker compose up -d
docker compose logs -f
docker compose down
```

## 📋 Code Review Process

### Review Checklist

- [ ] Code follows project style guidelines
- [ ] Functions and classes are well-documented
- [ ] Tests pass and coverage is maintained
- [ ] No breaking changes (or properly documented)
- [ ] Documentation is updated if needed
- [ ] Docker setup works correctly (if applicable)

### Review Timeline

- Initial review: Within 3-5 business days
- Follow-up reviews: Within 1-2 business days
- Merge decision: Based on code quality and project needs

## 📞 Getting Help

If you need help or have questions:

1. **Check Documentation**: Review the comprehensive documentation first
2. **Search Issues**: Look through existing issues and discussions
3. **Create Discussion**: Use GitHub Discussions for questions
4. **Contact Maintainers**: Reach out to core contributors for specific guidance

## 🎉 Recognition

Contributors will be recognized in:
- Project README
- Release notes
- GitHub contributors page
- Project documentation

Thank you for contributing to the NepaliGov-RAG-Bench project! Your contributions help make government information more accessible to citizens worldwide.

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (see LICENSE file for details).
