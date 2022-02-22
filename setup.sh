#!/usr/bin/env bash

git submodule update --init || exit 1
cd XED-to-XML
./mfile.py --opt=2 --no-encoder pymodule || exit 1
cp xed.* ..
cd ..
git submodule deinit -f --all
rm -rf .git/modules/*

wget https://www.uops.info/instructions.xml || exit 1
./convertXML.py instructions.xml || exit 1
rm instructions.xml
