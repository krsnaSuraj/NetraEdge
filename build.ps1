$env:JAVA_HOME = "F:\Android Studio\jbr"
$env:ANDROID_HOME = "F:\AndroidSDK"
$env:PATH = "$env:JAVA_HOME\bin;$env:PATH"

Set-Location "F:\PROJECTS\NetraEdge\packages\app\android"

Write-Host "========================================"
Write-Host " NetraEdge Android Build"
Write-Host "========================================"
Write-Host ""
Write-Host "JAVA_HOME: $env:JAVA_HOME"
Write-Host "ANDROID_HOME: $env:ANDROID_HOME"
Write-Host ""

Write-Host "Running Gradle build..."
& .\gradlew.bat assembleDebug --no-daemon

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "BUILD SUCCESSFUL!"
    Write-Host "APK: F:\PROJECTS\NetraEdge\packages\app\android\app\build\outputs\apk\debug\app-debug.apk"
} else {
    Write-Host ""
    Write-Host "BUILD FAILED - Check errors above"
}
