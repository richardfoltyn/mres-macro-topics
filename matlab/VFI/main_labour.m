%
% Topics in Macroeconomics (ECON5098), 2022-23
%
% Main file to run value function iteration (VFI) for problem with 
% deterministic labour income and plot results.
%
% Author: Richard Foltyn

% Add lib folder to search path
addpath('../lib')

% Always clear workspace at the beginning to make sure that we do not have
% any manually created variables in work space that might break our code.
clearvars

% close all figures
close all

%% Parameters

% Store all parameters in the struct variable par.

par.beta = 0.96;            % Discount factor
par.gamma = 2.0;            % Relative risk aversion (RRA)
par.r = 0.04;               % Interest rate (taken as given in part. eq.)
par.y = 1;                  % Constant labour income

% Asset grid parameters
par.a_min = 0.0;            % Lower bound of asset grid
par.a_max = 50;             % Upper bound of asset grid
par.N_a = 50;               % Number of points on asset grid

%% Asset grid

% We want to allocate more points towards the left end, i.e., at lower 
% asset levels.
% First create a grid of the desired shape on interval [0,1]
grid_01 = linspace(0.0, 1.0, par.N_a) .^ 1.3;
% Transform to final grid on interval [a_min, a_max]
grid_a = par.a_min + (par.a_max - par.a_min) .* grid_01;
% Store asset grid as column vector!
par.grid_a = grid_a';

%% Run VFI with grid search

% Termination tolerance for VFI
tol = 1.0e-6;
% Max. number of iterations
maxiter = 1000;

% Solve problem using grid search
[vfun, pfun_ia] = vfi(par, tol, maxiter);

% Savings policy function (optimal next-period asset level)
a_opt = par.grid_a(pfun_ia);

%% Run VFI with interpolation

% Solve problem using interpolation. Admissible interpolation methods
% are those accepted by the interp1() function, e.g. 'linear', 'cubic',
% 'pchip', 'spline', ...
% [vfun, a_opt] = vfi_interp(par, tol, maxiter, 'pchip');

%% Recover consumption policy function

% Cash-at-hand at beginning of period
cah = (1.0 + par.r) * par.grid_a + par.y;

% Consumption policy function (optimal consumption level)
pfun_cons = cah - a_opt;

%% Plot value and policy functions (simple plotting)

% Plot value functions
subplot(1,3,1);
plot(par.grid_a, vfun);
title('Value function');
xlabel('Assets');

% Plot savings (i.e. next-period assets)
subplot(1,3,2);
plot(par.grid_a, a_opt);
title('Savings');
xlabel('Assets');

% Plot optimal consumption
subplot(1,3,3);
plot(par.grid_a, pfun_cons);
title('Consumption');
xlabel('Assets');


%% Plot value and policy functions (advanced plotting)

% The code below creates prettier graphs which are otherwise identical
% to the simple graphs from above.

% Close previous figure
close all

% Settings governing plot style
aspect = 1.0;
% Limit plot region to part with more curvature
xlim_ = [0 5];
% Find largest index on asset grids to include in plots
ixmax = min(find(par.grid_a >= xlim_(2), 1), par.N_a);
xticks_ = linspace(0.0, xlim_(2), 6);
steelblue = [70/255, 130/255, 180/255];

% Plot value functions
subplot(1,3,1);
plot(par.grid_a, vfun, 'LineWidth', 1.5, 'Color', steelblue);
% Use custom function to set common plot style
set_ax_plot_style(gca, aspect);
title('Value function');
xlabel('Assets');
% Set plot limits and ticks
xlim(xlim_);
ylim([vfun(1), vfun(ixmax)]);
xticks(xticks_);

% Plot savings (i.e. next-period assets)
subplot(1,3,2);
plot(par.grid_a, a_opt, 'LineWidth', 1.5, 'Color', steelblue);
% Use custom function to set common plot style
set_ax_plot_style(gca, aspect);
title('Savings');
xlabel('Assets');
% Set plot limits and ticks
xlim(xlim_);
ylim([0, a_opt(ixmax)]);
xticks(xticks_);

% Plot optimal consumption
subplot(1,3,3);
plot(par.grid_a, pfun_cons, 'LineWidth', 1.5, 'Color', steelblue);
% Use custom function to set common plot style
set_ax_plot_style(gca, aspect);
title('Consumption');
xlabel('Assets');
% Set plot limits and ticks
xlim(xlim_);
ylim([pfun_cons(1), pfun_cons(ixmax)]);
xticks(xticks_);

%% Export figure to PDF

% Directory where graphs should be stored
graphdir = '../graphs';
% Create directory if it does not exist
mkdir(graphdir)

% Custom function to remove excessive margins around subplots
papersize = [8.5 3.0];
layout = [1 3];
tightlayout(gcf, layout, 0.02, papersize);

% Store graphs as PDF, add number of asset grid points to file name.
if exist('pfun_ia', 'var') ~= 0
    % Array pfun_ia is defined, so solution was obtained from grid search
    fn = sprintf('VFI_labour_N%d.pdf', par.N_a);
else
    % Solution was obtained using interpolation, append suffix
    fn = sprintf('VFI_labour_N%d_interp.pdf', par.N_a);
end

fn = fullfile(graphdir, fn);
saveas(gcf, fn)

