%
% Topics in Macroeconomics (ECON5098), 2020-21
%
% Main file to run value function iteration (VFI) for problem with deterministic
% labour and plot results.
%
% Author: Richard Foltyn

% Add lib/ folder to search path if it's not already at the beginning
addpath('./lib')

% Always clear workspace at the beginning to make sure that you do not have
% any manually created variables in work space that might break your code.
clearvars

% close all figures
close all

%% Parameters and grid setup
% Store all parameters in a struct

par.beta = 0.96;            % Discount factor
par.gamma = 2.0;            % Relative risk aversion (RRA)
par.r = 0.04;               % Interest rate
par.w = 1;                  % Wage rate

par.a_max = 5;              % Upper bound of asset grid
par.N_a = 50;               % Number of points on asset grid

% Asset grid: allocate more points towards the left end, i.e. at lower asset 
% levels.
grid_a = powerspace(0.0, par.a_max, par.N_a, 1.3);
% Store asset grid as column vector!
par.grid_a = grid_a';

%% Run value function iteration

[vfun, pfun_sav] = vfi(par);
% [vfun, pfun] = vfi_fast(par, grid, transm, futil);

% [vfun, pfun] = vfi_plot(par, grid, transm, futil);

%% Plot value and policy functions for savings and consumption

% Optimal next-period asset level
a_opt = par.grid_a(pfun_sav);

% Cash-at-hand at beginning of period
cah = (1.0 + par.r) * par.grid_a + par.w;
% Optimal consumption level
cons_opt = cah - a_opt;

% Plot value functions
subplot(1,3,1);
plot(par.grid_a, vfun, 'LineWidth', 1.5);
pbaspect([1.5 1 1]);
title('Value functions');
xlabel('Assets');

% Plot next-period assets
subplot(1,3,2);
plot(par.grid_a, a_opt, 'b', 'LineWidth', 1.5);
pbaspect([1.5 1 1]);
title('Optimal next-period assets');
xlabel('Assets');
set(gca, 'XLimSpec', 'Tight');
set(gca, 'YLimSpec', 'Tight');

% Plot optimal consumption
subplot(1,3,3);
plot(par.grid_a, cons_opt, 'LineWidth', 1.5);
pbaspect([1.5 1 1]);
title('Consumption');
xlabel('Assets');

%% Export figure to PDF
graphdir = 'graphs'
mkdir(graphdir)

set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [9 3.5]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 9 3.5]);


fn = sprintf("VFI_labour_N%d.pdf", par.N_a);
fn = fullfile(graphdir, fn)
saveas(gcf, fn)

