@echo off
:start

echo.
echo OVMS Chat Completion initiating...
echo.

set /p userInput="Please enter your message (type 'exit' or 'quit' to end): "

IF /I "%userInput%"=="exit" GOTO :eof
IF /I "%userInput%"=="quit" GOTO :eof

REM JSON 페이로드 내의 큰따옴표(")는 배치 스크립트에서 "" (두 개)로 이스케이프해야 합니다.
REM 긴 명령어를 여러 줄로 나누기 위해 ^ (캐럿)을 사용합니다.
REM IP address, port는 확인히세요
curl -s http://192.168.35.217:8001/v3/chat/completions ^
-H "Content-Type: application/json" ^
-d "{\"model\": \"LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct\", \"max_tokens\": 30, \"temperature\": 0, \"stream\": false, \"messages\": [{\"role\": \"system\", \"content\": \"You are a helpful assistant.\"}, {\"role\": \"user\", \"content\": \"%userInput%\"}]}"

echo.
echo Request completed
echo.
GOTO :start
