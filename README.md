# uiCA (uops.info Code Analyzer)

## Installation (Ubuntu)

Prerequisites:

    sudo apt-get install -y python3 python3-pip
    pip3 install plotly

Installation:

    git clone ...
    cd uiCA
    ./setup.sh


## Example Usage

	echo ".intel_syntax noprefix; add rax, rbx; add rbx, rax" > test.asm
    as test.asm -o test.o
    ./uiCA.py test.o -arch SKL
