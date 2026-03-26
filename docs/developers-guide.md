# Developer's Guide

## Building

Requires Go 1.25+.

Install from source:

```bash
go install github.com/harvard-cns/orla/cmd/orla@latest
```

Build from repo:

```bash
git clone https://github.com/harvard-cns/orla
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

## Python SDK (pyorla)

From the `pyorla` directory in the Orla repo:

```bash
uv sync
uv run ty check
uv run pytest
```

## Contributing

See [CONTRIBUTING.md](https://github.com/harvard-cns/orla/blob/main/CONTRIBUTING.md) in the Orla repository. Contributors are listed in [CONTRIBUTORS.md](https://github.com/harvard-cns/orla/blob/main/CONTRIBUTORS.md).
