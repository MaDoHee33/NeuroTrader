
import os
from pathlib import Path

# Configuration
SOURCE_DIRS = [Path("src"), Path("scripts"), Path("tests"), Path("config")]
DOC_FILES = [Path("README.md"), Path("DEV_JOURNAL.md"), Path("docs/ARCHITECTURE.md")]
OUTPUT_FILE = "NeuroTrader_Full_System_Export.md"

def main():
    print(f"Starting export to {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        # Header
        out.write("# NeuroTrader Full System Export\n")
        out.write(f"**Date:** {os.popen('date /t').read().strip()}\n")
        out.write("\n---\n\n")
        
        # 1. Documentation
        out.write("# 1. Documentation\n\n")
        for doc_path in DOC_FILES:
            if doc_path.exists():
                print(f"Adding Doc: {doc_path}")
                out.write(f"## File: {doc_path}\n")
                out.write("```markdown\n")
                try:
                    with open(doc_path, "r", encoding="utf-8") as f:
                        out.write(f.read())
                except Exception as e:
                    out.write(f"Error reading file: {e}")
                out.write("\n```\n\n")
            else:
                print(f"Skipping missing doc: {doc_path}")

        # 2. Source Code
        out.write("# 2. Source Code\n\n")
        file_count = 0
        
        for root_dir in SOURCE_DIRS:
            if not root_dir.exists():
                continue
                
            for root, dirs, files in os.walk(root_dir):
                # Skip cache and venv
                if "__pycache__" in root or ".venv" in root:
                    continue
                    
                for file in files:
                    if file.endswith((".py", ".yaml", ".json", ".sh", ".bat", ".ps1")):
                        file_path = Path(root) / file
                        print(f"Adding Code: {file_path}")
                        
                        out.write(f"## File: {file_path}\n")
                        
                        # Detect language
                        ext = file.split('.')[-1]
                        lang = "python" if ext == "py" else "yaml" if ext == "yaml" else "bash" if ext in ["sh", "bat", "ps1"] else ""
                        
                        out.write(f"```{lang}\n")
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                out.write(f.read())
                        except Exception as e:
                            out.write(f"Error reading file: {e}")
                        out.write("\n```\n\n")
                        file_count += 1
                        
        print(f"Export Complete! Total files: {file_count}")
        out.write(f"\n\n---\n**End of Export (Total Files: {file_count})**\n")

if __name__ == "__main__":
    main()
