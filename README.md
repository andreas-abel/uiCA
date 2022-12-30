# uiCA (uops.info Code Analyzer)

uiCA is a simulator that can predict the throughput of basic blocks on recent Intel microarchitectures.
In addition to that, it also provides insights into how the code is executed.

uiCA is based on data from [uops.info](https://www.uops.info), combined with a detailed pipeline model.
Like related tools, it assumes that all memory accesses result in cache hits.

Details on uiCA's pipeline model, as well as a comparison with similar tools, can be found in our paper [uiCA: Accurate Throughput Prediction of Basic Blocks on Recent Intel Microarchitectures](https://dl.acm.org/doi/pdf/10.1145/3524059.3532396).

## Web Interface

An online version of uiCA is available at [uiCA.uops.info](https://uiCA.uops.info).

## Installation
### Ubuntu

* Prerequisites:

      sudo apt-get install gcc python3 python3-pip graphviz
      pip3 install plotly

* Installation:

      git clone https://github.com/andreas-abel/uiCA.git
      cd uiCA
      ./setup.sh

* Update:

      git pull
      ./setup.sh

### Windows

* Prerequisites:
  * [Python 3](https://www.python.org/downloads/)
  * [MSVS compiler](https://visualstudio.microsoft.com/de/vs/features/cplusplus/)
  * `pip3 install plotly pydot`

* Installation:

      git clone https://github.com/andreas-abel/uiCA.git
      cd uiCA
      .\setup.cmd

* Update:

      git pull
      .\setup.cmd

## Example Usage

	echo ".intel_syntax noprefix; l: add rax, rbx; add rbx, rax; dec r15; jnz l" > test.asm
    as test.asm -o test.o
    ./uiCA.py test.o -arch SKL

## Command-Line Options

The following parameters are optional. Parameter names may be abbreviated if the abbreviation is unique (e.g., `-ar` may be used instead of `-arch`).

| Option                       | Description |
|------------------------------|-------------|
| `-arch`                  | The microarchitecture of the simulated CPU. Available microarchitectures: `SNB`, `IVB`, `HSW`, `BDW`, `SKL`, `SKX`, `KBL`, `CFL`, `CLX`, `ICL`, `TGL`, `RKL`. Alternatively, you can use `all` to get an overview of the throughputs for all supported microarchitectures.  `[Default: all]` |
| `-iacaMarkers`           | Analyze only the code that is between the `IACA_START` and `IACA_END` markers of Intel's [IACA](https://software.intel.com/content/www/us/en/develop/articles/intel-architecture-code-analyzer.html) tool. |
| `-raw`                   | Analyze a file that directly contains the machine code of the benchmark, but no headers or other data. |
| `-trace <filename.html>` | Generate an HTML file that contains a table with a cycle-by-cycle view of how the instructions are executed. |
| `-graph <filename.html>` | Generate an HTML file that contains a graph with various performance-related events.  |
| `-depGraph <filename.x>` | Output the dependency graph; the format is determined by the filename extension (e.g., svg, png, dot, etc.)  |
| `-alignmentOffset`       | Alignment offset (relative to a 64-Byte cache line). The option `all` provides an overview of the throughputs for all possible alignment offsets. `[Default: 0]` |
| `-TPonly`                | Output only the throughput prediction. |
| `-minIterations <n>`     | Simulate at least n iterations of the code. `[Default: 10]`. |
| `-minCycles <n>`         | Simulate at least n cycles. `[Default: 500]`. |
| `-simpleFrontEnd`        | Simulate a simple front end that is only limited by the issue width. |
| `-noMicroFusion`         | Simulate a CPU variant that does not support micro-fusion. |
| `-noMacroFusion`         | Simulate a CPU variant that does not support macro-fusion. |
