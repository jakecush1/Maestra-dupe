#!/bin/bash
# Double-click this file to run Parola on your Mac.
# It serves this folder on your own computer and opens it in your browser.
# Close the Terminal window (or press Ctrl+C) to stop.

cd "$(dirname "$0")" || exit 1

PORT=8000
while lsof -i ":$PORT" >/dev/null 2>&1; do PORT=$((PORT + 1)); done
URL="http://localhost:$PORT"

echo
echo "  Parola is running at $URL"
echo "  Leave this window open while you use it. Press Ctrl+C to stop."
echo

# Chrome gives the best speed; fall back to the default browser.
(sleep 1; open -a "Google Chrome" "$URL" 2>/dev/null || open "$URL") &

if command -v python3 >/dev/null 2>&1; then
  exec python3 -m http.server "$PORT" --bind 127.0.0.1
elif command -v ruby >/dev/null 2>&1; then
  exec ruby -run -e httpd . -b 127.0.0.1 -p "$PORT"
else
  echo "Could not find python3 or ruby to serve the page."
  echo "Install Python from https://www.python.org/downloads/ and try again."
  read -r -p "Press Return to close."
fi
