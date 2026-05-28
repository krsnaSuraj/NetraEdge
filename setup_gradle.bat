@echo off
set "JAVA_HOME=F:\Android Studio\jbr"
set "ANDROID_HOME=F:\AndroidSDK"
set "PATH=%JAVA_HOME%\bin;%PATH%"
cd /d "F:\PROJECTS\NetraEdge\packages\app\android"
"F:\.gradle\wrapper\dists\gradle-8.11.1-all\2qik7nd48slq1ooc2496ixf4i\gradle-8.11.1\bin\gradle.bat" wrapper --gradle-version 8.11.1 --distribution-type all
