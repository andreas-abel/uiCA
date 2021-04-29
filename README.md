# uiCA (uops.info Code Analyzer)

## Installation

    git clone ...    
    git submodule init
    git submodule update --recursive --remote
    cd XED-to-XML
    ./mfile.py examples
    cd ..
    wget https://www.uops.info/instructions.xml
    ./convertXML.py instructions.xml

## Example Usage

	echo ".intel_syntax noprefix; add rax, rbx; add rbx, rax" > test.asm
    as test.asm -o test.o
    ./uiCA.py test.o -arch SKL
