#!/usr/bin/env bash

git submodule update --init
cd XED-to-XML
./mfile.py examples
cp obj/wkit/bin/xed ..
cp disas.py .
cd ..
git submodule deinit --all
rm -rf .git/modules/*

wget https://www.uops.info/instructions.xml
./convertXML.py instructions.xml
