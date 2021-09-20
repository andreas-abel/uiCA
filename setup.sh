#!/usr/bin/env bash

git submodule update --init || exit 1
cd XED-to-XML
./mfile.py examples || exit 1
cp obj/wkit/bin/xed ..
cp disas.py ..
cd ..
git submodule deinit -f --all
rm -rf .git/modules/*

wget https://www.uops.info/instructions.xml || exit 1
./convertXML.py instructions.xml || exit 1
rm instructions.xml
