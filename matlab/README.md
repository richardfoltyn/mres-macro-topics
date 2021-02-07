
# MATLAB examples

## Description of directory structure

-   `VFI`: Solvers using value function iteration.
    -   [main_labour.m](VFI/main_labour.m): Solves the household problem with constant labour income
        and plots the solution.
        -   [vfi.m](VFI/vfi.m): VFI implementation
    -   [main_labour_risk.m](VFI/main_labour_risk.m): Solves the household problem
        with persistent labour income shocks and plots the solution.
        -   [vfi_risk.m](VFI/vfi_risk.m): VFI implementation with risky labour income.
-   `EGM`: Solvers using the endogenous grid-point method (TBA).
-   `lib`: Helper routines used throughout the code. 
-   `graphs`: Figures generated from code in the `VFI` and `EGM` folders.


## Requirements

The code has been successfully tested with
-   Matlab 2017b
-   [GNU Octave 5.2](https://www.gnu.org/software/octave/index)
