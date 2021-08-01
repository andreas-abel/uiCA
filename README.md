# uiCA (uops.info Code Analyzer)

uiCA is a simulator that can predict the throughput of basic blocks on recent Intel microarchitectures.
In addition to that, it also provides insights into how the code is executed.

uiCA is based on data from [uops.info](https://www.uops.info), combined with a detailed pipeline model.
Like related tools, it assumes that all memory accesses result in cache hits.

Details on uiCA's pipeline model, as well as a comparison with similar tools, can be found in our paper [Accurate Throughput Prediction of Basic Blocks on Recent Intel Microarchitectures](https://arxiv.org/pdf/2107.14210.pdf).

## Web Interface

An online version of uiCA is available at [uiCA.uops.info](https://uiCA.uops.info).

## Installation (Ubuntu)

Prerequisites:

    sudo apt-get install gcc python3 python3-pip
    pip3 install plotly

Installation:

    git clone ...
    cd uiCA
    ./setup.sh

## Installation (Windows)

Prerequisites: Python 3, Microsoft Visual C++ (might work with Clang installed as well)

For graphs:

    pip3 install plotly

Installation:

    git clone ...
    cd uiCA
    setup

## Example Usage

	echo ".intel_syntax noprefix; l: add rax, rbx; add rbx, rax; dec r15; jnz l" > test.asm
    as test.asm -o test.o
    ./uiCA.py test.o -arch SKL

## Command-Line Options

The following parameters are optional. Parameter names may be abbreviated if the abbreviation is unique (e.g., `-ar` may be used instead of `-arch`).

| Option                       | Description |
|------------------------------|-------------|
| `-arch`                  | The microarchitecture of the simulated CPU. Available options: `SNB`, `IVB`, `HSW`, `BDW`, `SKL`, `SKX`, `KBL`, `CFL`, `CLX`, `ICL`, `TGL`, `RKL`.  `[Default: SKL]` |
| `-iacaMarkers`           | Analyze only the code that is between the `IACA_START` and `IACA_END` markers of Intel's [IACA](https://software.intel.com/content/www/us/en/develop/articles/intel-architecture-code-analyzer.html) tool. |
| `-raw`                   | Analyze a file that directly contains the machine code of the benchmark, but no headers or other data. |
| `-trace <filename.html>` | Generate an HTML file that contains a table with a cycle-by-cycle view of how the instructions are executed. |
| `-graph <filename.html>` | Generate an HTML file that contains a graph with various performance-related events.  |
| `-alignmentOffset`       | Alignment offset (relative to a 64-Byte cache line). `[Default: 0]` |
| `-TPonly`                | Output only the throughput prediction. |
| `-simpleFrontEnd`        | Simulate a simple front end that is only limited by the issue width. |
| `-noMicroFusion`         | Simulate a CPU variant that does not support micro-fusion. |
| `-noMacroFusion`         | Simulate a CPU variant that does not support macro-fusion. |

Note that the IACA markers used here are slightly different from what the original IACA uses.
The start marker should be

    ud2 ; db 0fh, 0bh
    mov ebx, 111
    db 064h, 067h, 090h

and the end marker

    mov ebx, 222
    db 064h, 067h, 090h
    ud2 ; db 0fh, 0bh

(Stock IACA uses these sequences without UD2.)
