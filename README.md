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
  "reasoning_enabled": false,
  "replan_enabled": false,
  "replan_max_attempts": 1,
  "web_search_enabled": false,
  "web_search_engine": "searxng",
  "searxng_base_url": "http://127.0.0.1:8080",
  "searxng_preferred_engines": ["google"],
  "web_search_decision_use_image": false,
  "web_search_decision_reasoning_enabled": false,
  "web_search_decision_max_new_tokens": 64,
  "web_search_top_k": 5,
  "web_search_max_uses": 3,
  "web_search_timeout_s": 10,
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

추론 기능도 옵션으로만 켤 수 있습니다.
- 기본값: `reasoning_enabled = false`
- `reasoning_enabled = true` 또는 `--reasoning-enabled`
  - 모델이 다음 행동을 정할 때 추가 추론을 허용합니다.
  - 그래도 최종 출력 계약은 그대로 유지되며, agent는 실행 가능한 Python만 추출해서 executor로 보냅니다.

`replan_enabled`를 켜면 agent는 다음 조건에서 새 전략을 유도합니다.
- 직전과 같은 Python 코드가 다시 실행됨
- dependency repair 대상이 아닌 실행 오류가 발생함
- 최신 스크린샷 해시가 직전과 같아서 화면 변화가 없음

`replan_max_attempts`는 세션당 이 재계획 유도 신호를 몇 번까지 보낼지 정합니다.

웹서치도 옵션으로만 켤 수 있습니다.
- 기본값: `web_search_enabled = false`
- `web_search_enabled = true` 또는 `--web-search-enabled`
  - agent는 매 step마다 먼저 모델에게 `지금 웹서치가 필요한지`를 JSON으로 결정하게 합니다.
  - 검색 타이밍은 하드코딩 규칙이 아니라 모델이 현재 화면, 직전 실행 결과, replan 상태를 보고 정합니다.
  - 검색이 필요하다고 결정되면 `SearXNG` JSON API를 호출하고, 상위 결과를 다음 Python 생성 프롬프트에 `web_search_context`로 주입합니다.
- `web_search_engine`
  - 현재는 `searxng`만 지원합니다.
- `searxng_base_url`
  - 예: `http://127.0.0.1:8080`
- `searxng_preferred_engines`
  - 기본값은 `["google"]` 입니다.
  - agent는 먼저 `google` 엔진으로 검색하고, 결과가 비면 SearXNG 전체 엔진 결과로 fallback 합니다.
  - 같은 응답 안에 여러 엔진 결과가 섞여 있으면 `google` 결과를 앞쪽에 우선 배치합니다.
- `web_search_decision_use_image`
  - 기본값은 `false` 입니다.
  - 웹검색 필요 여부를 판단하는 decision 서브콜에서 스크린샷 이미지를 입력으로 쓰지 않습니다.
- `web_search_decision_reasoning_enabled`
  - 기본값은 `false` 입니다.
  - 웹검색 decision 서브콜에서는 reasoning을 별도로 끄고 더 가볍게 판단합니다.
- `web_search_decision_max_new_tokens`
  - 기본값은 `64` 입니다.
  - 웹검색 decision 서브콜 출력 길이 상한입니다.
- `web_search_top_k`
  - 모델 프롬프트에 넣을 검색 결과 개수입니다.
- `web_search_max_uses`
  - 세션당 실제 웹 검색 최대 횟수입니다.
- `web_search_timeout_s`
  - SearXNG 요청 timeout입니다.

예:

```bash
./.venv/bin/qwen-computer-use-agent \
  --model-id /home/kss930/model-projects/gui-owl-8B-think-1.0.0/models/Qwen3.5-9B \
  --endpoint http://127.0.0.1:8790 \
  --reasoning-enabled \
  --web-search-enabled \
  --searxng-base-url http://127.0.0.1:8080
```

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
