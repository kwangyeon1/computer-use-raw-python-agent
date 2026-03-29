# Agent Service Contract

## 역할

`computer-use-raw-python-agent`는 loop controller입니다.

- `--prompt`와 `--policy`를 직접 보유
- `--model-id`로 GUI-Owl base/merged model을 로드
- executor endpoint에서 현재 state를 가져옴
- 다음 step용 raw Python code를 만들어 executor에 보냄

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
  "screenshot_path": "C:/runs/current.png",
  "screenshot_base64": "<base64 png bytes>",
  "screenshot_media_type": "image/png",
  "observation_text": "Chrome bookmarks page is visible"
}
```
