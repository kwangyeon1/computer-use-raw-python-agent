# Agent Service Contract

## 역할

`computer-use-raw-python-agent`는 loop controller입니다.

- `--prompt`와 `--policy`를 직접 보유
- `--model-id` 또는 `--agent-cli-command`로 agent backend를 daemon에 로드
- executor endpoint에서 현재 state를 가져옴
- 다음 step용 raw Python code를 만들어 executor에 보냄

실행 형태는 두 단계입니다.
- 최초 1회: `computer-use-raw-python-agent --model-id ... --config config/agent.default.json`
- 또는: `computer-use-raw-python-agent --agent-cli-command ./.venv/bin/python /abs/path/to/wrapper.py --config config/agent.default.json`
- 또는 Codex 프로필: `computer-use-raw-python-agent --config config/agent.codex.gpt54.json`
- 이후 반복: `computer-use-raw-python-agent --prompt "..."`

## transport

- executor 연결:
  - `--endpoint` for HTTP
  - `--mcp-command` for stdio

## agent -> executor observe

```json
{
  "action": "observe"
}
```

## executor -> agent observe response

```json
{
  "ok": true,
  "action": "observe",
  "screenshot_path": "C:/runs/current.png",
  "screenshot_base64": "<base64 png bytes>",
  "screenshot_media_type": "image/png",
  "observation_text": "Chrome is open"
}
```

## agent model input

```json
{
  "user_prompt": "Chrome을 열고 북마크 관리자 페이지로 이동해줘",
  "policy": {
    "policy_name": "unrestricted_local"
  },
  "request_kind": "task_step",
  "repair_context": null,
  "replan_requested": true,
  "replan_reasons": [
    "repeated_code_execution",
    "no_visual_change"
  ],
  "strong_visual_grounding": false,
  "screenshot_path": "C:/runs/current.png",
  "screenshot_base64": "<base64 png bytes>",
  "screenshot_media_type": "image/png",
  "observation_text": "Chrome is open",
  "recent_history": ["step_0_return_code=0"],
  "last_execution": {
    "return_code": 0,
    "stdout_tail": "",
    "stderr_tail": ""
  },
  "step_index": 1
}
```

## agent -> external cli input

```json
{
  "action": "generate",
  "response_format": "python_code",
  "max_new_tokens": 256,
  "prompt_bundle": {
    "system_prompt": "...",
    "user_prompt": "...",
    "session_prompt": "Chrome을 열고 북마크 관리자 페이지로 이동해줘"
  },
  "rendered_prompt": "[system]\\n...\\n\\n[user]\\n...",
  "image_path": "C:/runs/current.png",
  "image_base64": null,
  "use_blank_image": false,
  "generation_context": {
    "run_dir": "C:/runs/bookmarks-loop",
    "step_id": "step-001",
    "request_kind": "task_step",
    "step_index": 1
  }
}
```

`agent.default.json`은 backend-neutral config이고, `agent_cli_command`는 command template이 아니라 argv 배열입니다. agent는 prompt를 `{prompt}`로 치환하지 않고, 위 JSON 전체를 stdin으로 보냅니다. package 내부 module wrapper를 실행할 때는 bare `python`보다 현재 venv Python 경로를 쓰는 편이 안전합니다.

repo 안의 `computer_use_raw_python_agent.codex_backend` wrapper를 쓰면 같은 `run_dir` 안의 step들은 `codex exec resume <session_id>`로 이어지고, 새 `run_dir`에서는 새 Codex session을 엽니다. 이 wrapper는 `screenshot`, `observation_text`, `last_execution`, `replan_reasons`는 유지하고, `recent_history`, `last_agent_response`, 반복된 `web_search_context`는 줄여서 Codex prompt를 구성합니다.

## external cli -> agent response

stdout는 두 방식 중 하나면 됩니다.

1. plain text 전체 출력
2. JSON 출력

```json
{
  "python_code": "print('hello')",
  "raw_text": "print('hello')",
  "model_id": "my-external-cli"
}
```

## agent -> executor execute

```json
{
  "action": "execute",
  "python_code": "focus_window(\"Chrome\")\npress_hotkey(\"ctrl\", \"l\")",
  "run_dir": "C:/runs/bookmarks-loop/steps/step-001",
  "step_id": "step-001",
  "metadata": {
    "step_index": 1
  }
}
```

## executor -> agent execute response

```json
{
  "ok": true,
  "action": "execute",
  "record": {
    "step_id": "step-001",
    "return_code": 0
  },
  "stdout_tail": "",
  "stderr_tail": "",
  "error_info": null,
  "screenshot_path": "C:/runs/current.png",
  "screenshot_base64": "<base64 png bytes>",
  "screenshot_media_type": "image/png",
  "observation_text": "Chrome bookmarks page is visible"
}
```
