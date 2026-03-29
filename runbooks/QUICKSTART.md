# Quickstart

```bash
cd /home/kss930/model-projects/gui-owl-8B-think-1.0.0/computer-use-raw-python-agent
python3 -m venv .venv
./.venv/bin/python -m pip install -e '.[dev]'
```

agent loop 실행:

```bash
./.venv/bin/python -m computer_use_raw_python_agent.service \
  --prompt "Chrome을 열고 북마크 관리자 페이지로 이동해줘" \
  --run-dir data/runs/bookmarks-loop \
  --mcp-command /home/kss930/model-projects/gui-owl-8B-think-1.0.0/computer-use-raw-python-executor/.venv/bin/python -m computer_use_raw_python_executor.cli --transport stdio --screenshot-path C:\\path\\to\\current_screen.png \
  --model-id /home/kss930/model-projects/gui-owl-8B-think-1.0.0/models/gui-owl-1.5-8b-think-base \
  --preload
```
