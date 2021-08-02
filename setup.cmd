@echo off
git submodule update --init
cd XED-to-XML
py mfile.py examples
copy /y obj\wkit\bin\xed.exe ..
copy /y disas.py ..
cd ..

git submodule deinit -f --all
rd /s /q .git\modules\*

PowerShell -Command Invoke-WebRequest https://www.uops.info/instructions.xml -OutFile instructions.xml
py convertXML.py instructions.xml
del instructions.xml
