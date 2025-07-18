@echo off
:start

echo.
echo OVMS Chat Completion initiating...
echo.

set /p userInput="Please enter your message (type 'exit' or 'quit' to end): "

IF /I "%userInput%"=="exit" GOTO :eof
IF /I "%userInput%"=="quit" GOTO :eof

echo Requesting...
for /f "tokens=1-4 delims=:." %%a in ("%time%") do set "startTime=%%a%%b%%c%%d"

REM JSON 페이로드 내의 큰따옴표(")는 배치 스크립트에서 "" (두 개)로 이스케이프해야 합니다.
REM 긴 명령어를 여러 줄로 나누기 위해 ^ (캐럿)을 사용합니다.
REM IP, Port 확인할것
curl -s http://192.168.35.217:8001/v3/chat/completions ^
-H "Content-Type: application/json" ^
-d "{\"model\": \"LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct\", \"max_tokens\": 30, \"temperature\": 0, \"stream\": false, \"messages\": [{\"role\": \"system\", \"content\": \"You are a helpful assistant.\"}, {\"role\": \"user\", \"content\": \"%userInput%\"}]}"

for /f "tokens=1-4 delims=:." %%a in ("%time%") do set "endTime=%%a%%b%%c%%d"

echo.
echo Request completed
echo.

call :calculateTimeDiff %startTime% %endTime%

echo.
GOTO :start

:calculateTimeDiff
set "start_ms=%1"
set "end_ms=%2"

REM 시, 분, 초, 1/100초를 밀리초로 변환
set /a "start_total_ms = ((1%start_ms:~0,2%-100)*3600000) + ((1%start_ms:~2,2%-100)*60000) + ((1%start_ms:~4,2%-100)*1000) + ((1%start_ms:~6,2%-100)*10)"
set /a "end_total_ms = ((1%end_ms:~0,2%-100)*3600000) + ((1%end_ms:~2,2%-100)*60000) + ((1%end_ms:~4,2%-100)*1000) + ((1%end_ms:~6,2%-100)*10)"

REM 자정을 넘어가거나 시작 시간이 종료 시간보다 늦을 경우 (이틀에 걸쳐 실행되는 경우) 처리
IF %end_total_ms% LSS %start_total_ms% SET /a "end_total_ms += 24 * 3600 * 1000"

set /a "time_diff_ms = end_total_ms - start_total_ms"

set /a "seconds = time_diff_ms / 1000"
set /a "milliseconds = time_diff_ms %% 1000"

echo Response time: %seconds%.%milliseconds% seconds
exit /b
