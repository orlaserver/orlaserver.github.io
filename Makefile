# Local preview for orlaserver.github.io (landing at /, Docsify at /docs/)
# Run `make` or `make help` for targets.

.DEFAULT_GOAL := help

PORT ?= 8080
HOST ?= 127.0.0.1
BASE := http://$(HOST):$(PORT)
PYTHON3 ?= python3

.PHONY: help serve smoke open legacy

help: ## Print available targets and variables
	@echo "Orla website — local preview"
	@echo ""
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-8s  %s\n", $$1, $$2}'
	@echo ""
	@echo "Variables: PORT=$(PORT) HOST=$(HOST)"

serve: ## Serve repo root at http://HOST:PORT/ (Ctrl+C to stop)
	@echo "Directory: $(CURDIR)"
	@echo "Landing:   $(BASE)/"
	@echo "Docs:      $(BASE)/docs/#/overview"
	@$(PYTHON3) -m http.server $(PORT) --bind $(HOST)

legacy: ## Static check: old links https://…/#/… redirect to /docs/#/… (no server; curl cannot test #)
	@test -f index.html || (echo "FAIL: run from website repo root"; exit 1)
	@grep -q "location.replace('/docs/' + h)" index.html || (echo "FAIL: legacy redirect (location.replace) missing"; exit 1)
	@grep -q "h.indexOf('#/') === 0" index.html || (echo "FAIL: legacy hash guard missing"; exit 1)
	@echo "legacy OK (script present; browser E2E: $(BASE)/#/quickstart → /docs/#/quickstart)"

smoke: ## Curl landing + docs paths (run make serve first; runs legacy checks first)
	@$(MAKE) -s legacy
	@echo "Checking $(BASE) ..."
	@curl -sf -o /dev/null -w "GET /                  -> %{http_code}\n" "$(BASE)/"
	@curl -sf -o /dev/null -w "GET /docs/overview.md  -> %{http_code}\n" "$(BASE)/docs/overview.md"
	@curl -sf -o /dev/null -w "GET /docs/index.html   -> %{http_code}\n" "$(BASE)/docs/index.html"
	@curl -sf "$(BASE)/" | grep -q "location.replace('/docs/'" || (echo "FAIL: root redirect script missing"; exit 1)
	@curl -sf "$(BASE)/docs/index.html" | grep -q "basePath" || (echo "FAIL: basePath missing in docs/index.html"; exit 1)
	@echo "smoke OK"

open: ## Open landing in default browser (macOS only)
	@open "$(BASE)/"
