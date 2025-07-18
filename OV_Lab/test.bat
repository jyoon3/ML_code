@echo off

echo.
echo OVMS Chat Completion initiating...
echo.

REM JSON 페이로드 내의 큰따옴표(")는 배치 스크립트에서 "" (두 개)로 이스케이프해야 합니다.
REM 긴 명령어를 여러 줄로 나누기 위해 ^ (캐럿)을 사용합니다.
curl -s http://localhost:8000/v3/chat/completions ^
-H "Content-Type: application/json" ^
-d "{\"model\": \"\"OpenVINO/Phi-3.5-mini-instruct-int4-ov\"\", \"max_tokens\": 30, \"temperature\": 0, \"stream\": false, \"messages\": [{\"role\": \"system\", \"content\": \"\"You are a helpful assistant.\"\"}, {\"role\": \"user\", \"content\": \"\"What are the 3 main tourist attractions in Paris?\"\"}]}"

echo.
echo Request completed
echo.
pause