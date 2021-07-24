# uiCA (uops.info Code Analyzer)

## Installation (Linux)

    git clone ...
    cd uiCA
    ./setup.sh


## Example Usage

	echo ".intel_syntax noprefix; add rax, rbx; add rbx, rax" > test.asm
    as test.asm -o test.o
    ./uiCA.py test.o -arch SKL
