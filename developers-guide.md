# Developer's Guide

## Building

Requires Go 1.25+.

**Install from source:**

```bash
go install github.com/dorcha-inc/orla/cmd/orla@latest
```

**Build from repo:**

```bash
git clone https://github.com/dorcha-inc/orla
cd orla
make build
make install
```

## Testing

Run the test suite:

```bash
make test
```

Integration tests:

```bash
make test-integration
```

## Contributing

See [CONTRIBUTING.md](https://github.com/dorcha-inc/orla/blob/main/CONTRIBUTING.md) in the Orla repository. Contributors are listed in [CONTRIBUTORS.md](https://github.com/dorcha-inc/orla/blob/main/CONTRIBUTORS.md).
