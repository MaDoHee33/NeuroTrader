---
name: Watcher Skill
description: The "Eyes" of NeuroTrader. Uses visual browsing to extract intelligence from news sites, social media, and research reports.
---

# Watcher Skill 👁️

The Watcher connects `NeuroTrader` to the visual web using the `browser` CLI agent. It bypasses API limitations by traversing websites directly, just like a human analyst.

## Capabilities
- **Deep Extraction**: Reads full article content, not just headlines.
- **Visual Context**: Can see charts and images (future upgrade).
- **Stealth**: Uses browser automation to access public information securely.

## Components
- **`watcher.py`**: Main controller script.
    - `fetch(url)`: Visits a URL and extracts content as JSON.
    - `scan(targets)`: Iterates through a list of target URLs.
- **`dump/`**: Storage for raw extracted data.

## Usage
```bash
# Fetch single URL
python skills/watcher/watcher.py --url "https://cryptopanic.com/news/gold"

# Run Monitor Loop
python skills/watcher/watcher.py --monitor
```

## Dependencies
- `camhahu/browser` CLI (installed via `install.sh`)
- `python >= 3.10`
