# Raw Python Agent Profile

이 repo는 `GUI-Owl` 계열 모델이 unrestricted local Python을 생성하는 구조를 위한 설계/실행 repo입니다.

## 목표

- prompt와 executor 상태를 보고 Python GUI/OS 코드 생성
- executor 종속 action schema를 제거
- agent가 loop를 소유하고 executor는 실행만 수행
- 학습 데이터 포맷을 `screen -> python code`로 고정

## 비목표

- 실행 환경 격리
- 복구 가능한 샌드박스
- destructive capability 제거
- executor-specific JSON action tuning

## output contract

모델 출력은 `raw Python`입니다.

예:

```python
focus_window("Chrome")
press_hotkey("ctrl", "l")
switch_input_locale("en-US")
type_text("chrome://bookmarks")
press_key("enter")
```

또는 직접 라이브러리 호출도 허용합니다.

```python
import pyautogui
import time

pyautogui.hotkey("ctrl", "l")
time.sleep(0.3)
pyautogui.write("chrome://bookmarks", interval=0.01)
pyautogui.press("enter")
```

## 운영 원칙

1. generated Python 원문은 항상 저장
2. stdout/stderr는 항상 저장
3. run metadata는 항상 저장
4. unrestricted 실행 여부는 policy file에 명시
5. executor-specific schema가 아니라 code semantics를 학습
6. prompt/policy는 agent가 보유

## repo 계보

- source inspiration:
  - `../computer-use-stage1-actor`
  - `../computer-use-stage1-base-probe`
- runtime/output philosophy:
  - executor-independent raw Python

## 권장 다음 단계

1. raw Python prompt 고정
2. Windows runner helper API 정의
3. sample task들로 codegen 수집
4. `screen + prompt + history -> python code` 데이터셋 생성
5. GUI-Owl LoRA 학습
