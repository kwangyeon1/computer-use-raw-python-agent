# computer-use-raw-python-agent

`GUI-Owl` 같은 비전 모델을 로드하고, executor endpoint에 붙어서
`현재 화면 + 사용자 프롬프트 + 직전 실행 결과`를 보고 다음 `raw Python` 코드를 생성하는 최소 agent repo입니다.

- 역할: loop controller + vision-conditioned raw Python code generator
- 기본 모델: `../models/gui-owl-1.5-8b-think-base`
- 입력: `--prompt`, `--policy`, executor state
- 출력: next-step Python 코드와 loop artifact
- executor 연결: `--endpoint` 또는 `--mcp-command`

## 핵심 구조

```text
agent owns prompt and loop
  -> observe current state from executor
  -> model generates raw python
  -> send python code to executor
  -> executor runs code locally and returns next state
  -> repeat
```

이 repo가 실제 loop driver입니다.

## 빠른 시작

```bash
cd /home/kss930/model-projects/gui-owl-8B-think-1.0.0/computer-use-raw-python-agent

./.venv/bin/python -m computer_use_raw_python_agent.service \
  --prompt "Chrome을 열고 북마크 관리자 페이지로 이동해줘" \
  --run-dir data/runs/bookmarks-loop \
  --mcp-command /home/kss930/model-projects/gui-owl-8B-think-1.0.0/computer-use-raw-python-executor/.venv/bin/python -m computer_use_raw_python_executor.cli --transport stdio --screenshot-path C:\\path\\to\\current_screen.png \
  --model-id /home/kss930/model-projects/gui-owl-8B-think-1.0.0/models/gui-owl-1.5-8b-think-base \
  --preload
```

## 남긴 파일

- `src/computer_use_raw_python_agent/service.py`
- `src/computer_use_raw_python_agent/executor_client.py`
- `src/computer_use_raw_python_agent/runtime.py`
- `src/computer_use_raw_python_agent/prompting.py`
- `src/computer_use_raw_python_agent/models.py`
- `config/policy.unrestricted.local.json`
- `docs/AGENT_SERVICE_CONTRACT.md`

executor state는 `screenshot_path`만이 아니라 transport 내부 `screenshot_base64`도 포함할 수 있고,
agent는 그 이미지를 내부적으로 디코드해서 바로 추론에 사용합니다.

## 제거한 것

- one-shot prompt CLI
- prompt 렌더링 CLI
- agent 내부 executor 코드
- 별도 구조화 지시 스펙 입력 경로
