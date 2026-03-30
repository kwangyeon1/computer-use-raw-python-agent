# Quickstart

```bash
cd /home/kss930/model-projects/gui-owl-8B-think-1.0.0/computer-use-raw-python-agent
python3 -m venv .venv
./.venv/bin/python -m pip install -e '.[dev]'
```

agent loop 실행:

```bash
./.venv/bin/computer-use-raw-python-agent \
  --model-id /home/kss930/model-projects/gui-owl-8B-think-1.0.0/models/gui-owl-1.5-8b-think-base \
  --config config/agent.default.json
```

prompt 실행:

```bash
./.venv/bin/computer-use-raw-python-agent \
  --prompt "Chrome을 열고 북마크 관리자 페이지로 이동해줘"
```
