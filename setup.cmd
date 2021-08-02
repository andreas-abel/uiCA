@echo off
git submodule update --init || exit /b
cd XED-to-XML
py mfile.py examples || exit /b
copy /y obj\wkit\bin\xed.exe ..
copy /y disas.py ..
cd ..

git submodule deinit -f --all
rd /s /q .git\modules\*

PowerShell -Command Invoke-WebRequest https://www.uops.info/instructions.xml -OutFile instructions.xml || exit /b
py convertXML.py instructions.xml || exit /b
del instructions.xml
