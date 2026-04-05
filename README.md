# computer-use-raw-python-agent

`GUI-Owl` 같은 비전 모델을 한 번 로드한 뒤 daemon 형태로 유지하고,
executor endpoint에 붙어서 `현재 화면 + 사용자 프롬프트 + 직전 실행 결과`를 보고
다음 `raw Python` 코드를 생성하는 최소 agent repo입니다.

- 역할: loop controller + vision-conditioned raw Python code generator
- 기본 모델: `../models/gui-owl-1.5-8b-think-base`
- 입력: `--prompt`, config JSON, executor state
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

./.venv/bin/computer-use-raw-python-agent \
  --model-id /home/kss930/model-projects/gui-owl-8B-think-1.0.0/models/gui-owl-1.5-8b-think-base \
  --config config/agent.default.json
```

이후에는 모델이 daemon에 남아 있으므로 prompt만 보내면 됩니다.

```bash
./.venv/bin/computer-use-raw-python-agent \
  --prompt "Chrome을 열고 북마크 관리자 페이지로 이동해줘"
```

첫 `--model-id` 로딩은 시간이 오래 걸릴 수 있습니다. 그래서 timeout은 둘로 나뉩니다.
- model load/reload: `load_request_timeout_s`
- prompt 실행: `run_request_timeout_s`

각각 CLI에서 `--load-request-timeout-s`, `--run-request-timeout-s`로 따로 덮어쓸 수 있습니다.

기본 config 예시는 [config/agent.default.json](/home/kss930/model-projects/gui-owl-8B-think-1.0.0/computer-use-raw-python-agent/config/agent.default.json) 입니다.

```json
{
  "endpoint": "http://127.0.0.1:8790",
  "policy": "policy.unrestricted.local.json",
  "run_dir": "../data/runs",
  "max_iterations": 5,
  "max_new_tokens": 256,
  "strong_visual_grounding": false,
  "replan_enabled": false,
  "replan_max_attempts": 1,
  "dependency_repair_enabled": false,
  "dependency_repair_max_attempts": 2,
  "dependency_repair_allow_shell_fallback": false,
  "load_request_timeout_s": 120,
  "run_request_timeout_s": 100
}
```

시각 grounding 강제는 옵션으로만 켤 수 있습니다.
- 기본값: `strong_visual_grounding = false`
- `strong_visual_grounding = true` 또는 `--strong-visual-grounding`
  - system/user prompt에 현재 스크린샷을 1차 근거로 사용하라는 규칙을 추가합니다.
  - 같은 사용자 요청만 반복적으로 따르기보다, 현재 화면 변화와 최근 실행 결과를 더 강하게 반영하도록 유도합니다.

`replan_enabled`를 켜면 agent는 다음 조건에서 새 전략을 유도합니다.
- 직전과 같은 Python 코드가 다시 실행됨
- dependency repair 대상이 아닌 실행 오류가 발생함
- 최신 스크린샷 해시가 직전과 같아서 화면 변화가 없음

`replan_max_attempts`는 세션당 이 재계획 유도 신호를 몇 번까지 보낼지 정합니다.

의존성 자동 복구도 옵션으로만 켤 수 있습니다.
- 기본값: `dependency_repair_enabled = false`
- `dependency_repair_enabled = true` 또는 `--dependency-repair-enabled`
  - executor가 `ModuleNotFoundError: No module named '...'`를 정확히 분류했을 때만 복구 모드를 탑니다.
  - agent는 먼저 pip 기반 복구 코드를 생성합니다.
  - 복구 성공 시 원래 작업 코드를 1회 자동 재실행합니다.
- `dependency_repair_max_attempts`
  - 세션당 자동 복구 최대 시도 횟수입니다.
- `dependency_repair_allow_shell_fallback = true` 또는 `--dependency-repair-allow-shell-fallback`
  - pip 기반 복구 뒤에도 필요하면 subprocess 기반 shell/batch 설치 경로를 한 번 더 시도할 수 있게 합니다.

## Qwen Variants

`transformers` 기반 Qwen 경로는 기존대로 [qwen-computer-use-agent](/home/kss930/model-projects/gui-owl-8B-think-1.0.0/computer-use-raw-python-agent/pyproject.toml) 를 사용합니다.

SGLang으로 같은 agent loop / executor I/O 계약을 유지하면서 모델 서빙을 분리하려면 `qwen-sglang`를 사용합니다.

이 경로는 agent와 같은 Python 환경에서 `python -m sglang.launch_server`를 실행하므로, `sglang` 패키지가 해당 `.venv`에 설치되어 있어야 합니다.

```bash
./.venv/bin/qwen-sglang \
  --model-id /home/kss930/model-projects/gui-owl-8B-think-1.0.0/models/Qwen3.5-9B \
  --endpoint http://127.0.0.1:8790
```

이 경로는 별도 daemon/runtime 파일을 사용합니다.
- `src/computer_use_raw_python_agent/sglang_cli.py`
- `src/computer_use_raw_python_agent/sglang_daemon.py`
- `src/computer_use_raw_python_agent/sglang_runtime.py`
- `src/computer_use_raw_python_agent/sglang_config.py`

기본 설정은 [config/agent.qwen35.sglang.default.json](/home/kss930/model-projects/gui-owl-8B-think-1.0.0/computer-use-raw-python-agent/config/agent.qwen35.sglang.default.json) 에 있습니다.

SGLang variant의 모델/서버 설정은 `transformers` 경로와 다릅니다.
- `sglang_server_host`
- `sglang_server_port`
- `sglang_server_ready_timeout_s`
- `sglang_request_timeout_s`
- `sglang_dtype`
- `sglang_tp_size`
- `sglang_mem_fraction_static`
- `sglang_server_extra_args`

하지만 executor에 주고받는 입력/출력 계약은 기존 Qwen agent와 동일합니다.

Qwen 경로와 SGLang 경로 모두 `reasoning_enabled`를 지원합니다.
- 기본값: `false`
- CLI: `--reasoning-enabled`
- config: `"reasoning_enabled": true`

이 옵션은 모델이 내부 추론을 사용할 수 있게 허용하지만, 최종 출력 계약은 그대로 `executable Python only`입니다.

## 남긴 파일

- `src/computer_use_raw_python_agent/cli.py`
- `src/computer_use_raw_python_agent/daemon.py`
- `src/computer_use_raw_python_agent/config_utils.py`
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
