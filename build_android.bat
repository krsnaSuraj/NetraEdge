@echo off
echo ========================================
echo  NetraEdge Android Build
echo ========================================
echo.

set "JAVA_HOME=F:\Android Studio\jbr"
set "ANDROID_HOME=F:\AndroidSDK"
set "PATH=%JAVA_HOME%\bin;%ANDROID_HOME%\platform-tools;%PATH%"

echo [1/3] Setting up environment...
echo   JAVA_HOME: %JAVA_HOME%
echo   ANDROID_HOME: %ANDROID_HOME%
echo.

echo [2/3] Building Android app...
cd /d "F:\PROJECTS\NetraEdge\packages\app\android"
call gradlew.bat assembleDebug --no-daemon
if %ERRORLEVEL% neq 0 (
    echo.
    echo BUILD FAILED
    echo Check the error messages above.
    pause
    exit /b 1
)

echo.
echo [3/3] Build complete!
echo APK location: F:\PROJECTS\NetraEdge\packages\app\android\app\build\outputs\apk\debug\app-debug.apk
echo.
pause
