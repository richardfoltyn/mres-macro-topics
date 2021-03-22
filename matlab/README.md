
# MATLAB examples

## Description of directory structure

-   `VFI`: Solvers using value function iteration.
    -   [main_labour.m](VFI/main_labour.m): Solves the household problem with constant labour income
        and plots the solution.
        -   [vfi.m](VFI/vfi.m): VFI implementation
    -   [main_labour_risk.m](VFI/main_labour_risk.m): Solves the household problem
        with persistent labour income shocks and plots the solution.
        -   [vfi_risk.m](VFI/vfi_risk.m): VFI implementation with risky labour income.
        -   [vfi_risk_howard.m](VFI/vfi_risk_howard.m): VFI implementation with
            risky labour income. This is an extension of plain VFI that uses
            Howard's improvement algorithm to accelerate convergence. 
-   `EGM`: Solvers using the endogenous grid-point method.
    -   [main_labour.m](EGM/main_labour.m): Solves the household problem with constant labour income
        and plots the solution.
        -   [egm_IH.m](EGM/egm_IH.m): EGM implementation (infinite horizon)
    -   [main_labour_risk.m](EGM/main_labour_risk.m): Solves the household problem
        with persistent labour income shocks and plots the solution.
        -   [egm_IH_risk.m](EGM/egm_IH_risk.m): EGM implementation with risky labour (infinite horizon)
    -   [compare.mn](EGM/compare.m): Create plots to compute household
            policy functions for two different parametrisations.
-   `lib`: Helper routines used throughout the code. 
-   `graphs`: Figures generated from code in the `VFI` and `EGM` folders.


## Requirements

The code has been successfully tested with
-   Matlab 2017b
-   [GNU Octave 5.2](https://www.gnu.org/software/octave/index)
