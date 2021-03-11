%
% Topics in Macroeconomics (ECON5098), 2020-21
%
% Main file to run value function iteration (VFI) for problem with 
% risky labour income and plot results.
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
par.r = 0.04;               % Interest rate

% Asset grid parameters
par.a_max = 50;             % Upper bound of asset grid
par.N_a = 50;               % Number of points on asset grid

% Labour income process parameters
par.rho = 0.95;             % Persistence of AR(1)
par.sigma = 0.20;           % Conditional std. dev. of AR(1)
par.N_y = 3;                % Grid size for discretised process

%% Grids

% Asset grid: allocate more points towards the left end, i.e. at lower asset 
% levels.
grid_a = powerspace(0.0, par.a_max, par.N_a, 1.3);
% Store asset grid as column vector!
par.grid_a = grid_a';


% Discretize AR(1) labour income process to a first-order Markov chain
mu = 0.0;                   % Mean of AR(1)

[states, tm_y] = rouwenhorst(par.N_y, mu, par.rho, par.sigma);
% Compute ergodic distribution of Markov approximation
edist = markov_ergodic_dist(tm_y);
% Convert state space from logs to levels
states = exp(states);
% Normalise states such that unconditional expectation is 1.0
grid_y = states / (edist' * states);

% Store grid and transition matrix in par object
par.grid_y = grid_y;
par.tm_y = tm_y;


%% Run value function iteration

% Termination tolerance for VFI
tol = 1.0e-6;
% Max. number of iterations
maxiter = 1000;

% Solve problem using grid search
[vfun, pfun_ia] = vfi_risk(par, tol, maxiter);

% Run VFI with Howard's improvement acceleration algorithm instead.
% This is an extension of plain VFI. You can ignore this algorithm
% initially.
% naccel = 10;
% [vfun, pfun_ia] = vfi_risk_howard(par, tol, maxiter,naccel);

% Optimal next-period asset level (savings)
a_opt = par.grid_a(pfun_ia);

% Solve problem using interpolation. Admissible interpolation methods
% are those accepted by the interp1() function, e.g. 'linear', 'cubic',
% 'pchip', 'spline', ...
% [vfun, a_opt] = vfi_risk_interp(par, tol, maxiter, 'pchip');


%% Plot value and policy functions for savings and consumption

% Cash-at-hand at beginning of period
cah = (1.0 + par.r) * par.grid_a + par.grid_y';

% Optimal consumption level
cons_opt = cah - a_opt;

% Settings governing plot style
aspect = 1.0;
% Limit plot region to part with more curvature
xlim_ = [0 5];
% Find largest index on asset grids to include in plots
ixmax = min(find(par.grid_a >= xlim_(2), 1), par.N_a);
xticks_ = linspace(0.0, xlim_(2), 6);

red = [228/255, 26/255, 28/255];
steelblue = [70/255, 130/255, 180/255];
green = [77/255 175/255 74/255]; 

% Add transparency to colour codes in MATLAB. This is not supported in Octave.
if ~isOctave()
    low = [red 0.8];
    mid = [steelblue 0.8];
    high = [green 0.8];
else
    low = red; mid = steelblue; high = green;
end

colours = [low' mid' high'];

% Legend labels
labels = arrayfun(@(x) sprintf('y=%.3f', x), par.grid_y, 'UniformOutput', false);

% Plot value functions
subplot(1,3,1);
for iy = 1:par.N_y
    plot(par.grid_a, vfun(:,iy), 'LineWidth', 1.5, 'Color', colours(:,iy));
    hold on;
end
hold off;
% Use custom function to set common plot style
set_ax_plot_style(gca, aspect);
title('Value function');
xlabel('Assets');
% Set plot limits and ticks
xlim(xlim_);
ylim([min(vfun(1,:)), max(vfun(ixmax,:))]);
xticks(xticks_);
legend(labels, 'Location', 'SouthEast');

% Plot savings (i.e. next-period assets)
subplot(1,3,2);
for iy = 1:par.N_y
    plot(par.grid_a, a_opt(:,iy), 'LineWidth', 1.5, 'Color', colours(:,iy));
    hold on;
end
hold off;
% Use custom function to set common plot style
set_ax_plot_style(gca, aspect);
title('Savings');
xlabel('Assets');
% Set plot limits and ticks
xlim(xlim_);
ylim([0, max(a_opt(ixmax,:))]);
xticks(xticks_);

% Plot optimal consumption
subplot(1,3,3);
for iy = 1:par.N_y
    plot(par.grid_a, cons_opt(:,iy), 'LineWidth', 1.5, 'Color', colours(:,iy));
    hold on;
end
hold off;
% Use custom function to set common plot style
set_ax_plot_style(gca, aspect);
title('Consumption');
xlabel('Assets');
% Set plot limits and ticks
xlim(xlim_);
ylim([min(cons_opt(1,:)), max(cons_opt(ixmax,:))]);
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
    fn = sprintf('VFI_labour_risk_N%d.pdf', par.N_a);
else
    % Solution was obtained using interpolation, append suffix
    fn = sprintf('VFI_labour_risk_N%d_interp.pdf', par.N_a);
end

fn = fullfile(graphdir, fn);
saveas(gcf, fn)

