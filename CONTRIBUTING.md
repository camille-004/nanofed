# 🤝 Contributing to NanoFed

---

## 🚀 Getting Started

### Development Setup

1. **Fork & Clone**
    ```bash
    git clone https://github.com/your-username/nanofed.git
    cd nanofed
    ```

2. **Set Up Environment**
    ```bash
    make install
    poetry shell
    ```

3. **Create Feature Branch**
    ```bash
    git checkout -b feature/your-feature-name
    ```

---

## 📝 Commit Guidelines

```bash
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

| Type       | Description              |
| :--------- | :----------------------- |
| `feat`     | New features             |
| `fix`      | Bug fixes                |
| `docs`     | Documentation            |
| `style`    | Code style/formatting    |
| `refactor` | Code refactoring         |
| `perf`     | Performance improvements |
| `test`     | Adding/updating tests    |
| `build`    | Build system changes     |
| `ci`       | CI changes               |
| `chore`    | Maintenance tasks        |

### Best Practices

- Use imperative mood ("add" not "added")
- Don't capitalize first letter
- No period at the end
- Keep description under 72 characters
- Use body for "what" and "why"

---

## 🔄 Pull Request Process

### Steps

1. Update documentation
2. Add unit or integration tests for new features
3. Make sure all tests pass
4. Target the `main` branch
5. 👀 Get at least one review

### PR Title Format

Follow the commit message convention:

```bash
feat(client): add new automatic feature
```
