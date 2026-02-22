# Uninstalling Orla

**Homebrew:**

```bash
brew uninstall --cask orla
```

**Install script:**

```bash
curl -fsSL https://raw.githubusercontent.com/dorcha-inc/orla/main/scripts/uninstall.sh | sh
```

The uninstall script removes only Orla. Ollama and downloaded models are left in place. To remove Ollama:

- **Homebrew:** `brew uninstall ollama`
- **Other:** See [ollama.ai](https://ollama.ai) or your system’s package manager.
