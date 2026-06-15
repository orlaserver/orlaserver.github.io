# Local preview for orlaserver.github.io.
# The landing page is at /, the docsify docs at /docs/.

host := "127.0.0.1"
port := "8080"
base := "http://" + host + ":" + port

# List available recipes.
default:
    @just --list

# Serve the site locally. Ctrl+C to stop. Override with `just port=9000 serve`.
serve:
    @echo "landing  {{ base }}/"
    @echo "docs     {{ base }}/docs/"
    python3 -m http.server {{ port }} --bind {{ host }}

# Open the landing page in a browser (macOS).
open:
    open "{{ base }}/"

# Check the key pages respond. Run `just serve` in another terminal first.
check:
    curl -sf -o /dev/null -w "/                      %{http_code}\n" "{{ base }}/"
    curl -sf -o /dev/null -w "/docs/index.html       %{http_code}\n" "{{ base }}/docs/index.html"
    curl -sf -o /dev/null -w "/docs/v2/quickstart.md %{http_code}\n" "{{ base }}/docs/v2/quickstart.md"
