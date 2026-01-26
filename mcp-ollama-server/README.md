# MCP Server for Ollama

MCP (Model Context Protocol) Server ที่เชื่อมต่อกับ Ollama API สำหรับใช้งานโมเดล AI ในเครื่อง

## ⚡ Quick Start

### 1. Install Dependencies
```powershell
cd C:\Users\pp\.gemini\antigravity\scratch\mcp-ollama-server
pip install -r requirements.txt
```

### 2. Start Ollama Server
ตรวจสอบว่า Ollama กำลังรันอยู่:
```powershell
ollama list
```

### 3. Run MCP Server
```powershell
python server.py
```

---

## 🔧 Configuration for Claude Desktop

เพิ่มใน `%APPDATA%\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ollama": {
      "command": "python",
      "args": ["C:\\Users\\pp\\.gemini\\antigravity\\scratch\\mcp-ollama-server\\server.py"]
    }
  }
}
```

---

## 🛠️ Available Tools

| Tool | Description |
|------|-------------|
| `ask_deepseek` | ส่งคำถามไปยัง DeepSeek model |
| `ask_with_context` | ส่งคำถามพร้อม context (code, document) |
| `list_ollama_models` | แสดงรายการโมเดลที่มี |
| `check_ollama_status` | ตรวจสอบสถานะ Ollama server |

---

## 📝 Notes

- Default model: `deepseek-v3.1:671b-cloud`
- Ollama API: `http://localhost:11434`
- Timeout: 5 minutes สำหรับ generation
