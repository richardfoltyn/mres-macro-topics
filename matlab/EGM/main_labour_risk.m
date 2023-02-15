%
% Topics in Macroeconomics (ECON5098), 2022-23
%
% Main file to run the infinite-horizon endogenous grid-point method (EGM) 
% for problem with risky labour income and plot results.
%
% Author: Richard Foltyn

% Add lib folder to search path
addpath('../lib')

% Always clear workspace at the beginning to make sure that we do not have
% any manually created variables in workspace that might break our code.
clearvars

% close all figures
close all

%% Parameters

% Store all parameters in the struct variable par.

par.beta = 0.96;            % Discount factor
par.gamma = 2.0;            % Relative risk aversion (RRA)
par.r = 0.04;               % Interest rate

% Asset grid parameters
par.a_min = 0;              % Lower bound of asset grid
par.a_max = 50;             % Upper bound of asset grid
par.N_a = 100;              % Number of points on asset grid

% Labour income process parameters
par.rho = 0.95;             % Persistence of AR(1)
par.sigma = 0.20;           % Conditional std. dev. of AR(1)
par.N_y = 3;                % Grid size for discretised process

%% Asset and savings grids

% Note: we use the same grid for assets and savings, since
%   savings = a'
%   a' becomes the initial asset position next period.

% We want to allocate more points towards the left end, i.e., at lower 
% asset levels.
% First create a grid of the desired shape on interval [0,1]
grid_01 = linspace(0.0, 1.0, par.N_a) .^ 1.3;
% Transform to final grid on interval [a_min, a_max]
grid_a = par.a_min + (par.a_max - par.a_min) .* grid_01;

% Store asset grid as column vector!
par.grid_a = grid_a';


%% Labour income grid

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

%% Run EGM

% Termination tolerance for infinite-horizon EGM
tol = 1.0e-6;
% Max. number of iterations
maxiter = 1000;

% Solve problem using infinite-horizon EGM algorithm.
% Function returns consumption and savings policy functions defined on
% asset grid.
[pfun_cons, pfun_sav] = egm_IH_risk(par, tol, maxiter);

%% Plot policy functions for savings and consumption (simple plots)


% Plot savings (i.e. next-period assets)
subplot(1,2,1);
plot(par.grid_a, pfun_sav);
title('Savings');
xlabel('Assets');


% Plot optimal consumption
subplot(1,2,2);
plot(par.grid_a, pfun_cons);
title('Consumption');
xlabel('Assets');

%% Plot policy functions for savings and consumption (advanced plots)


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

% Plot savings (i.e. next-period assets)
subplot(1,2,1);
for iy = 1:par.N_y
    plot(par.grid_a, pfun_sav(:,iy), 'LineWidth', 1.5, 'Color', colours(:,iy));
    hold on;
end
hold off;
% Use custom function to set common plot style
set_ax_plot_style(gca, aspect);
title('Savings');
xlabel('Assets');
% Set plot limits and ticks
xlim(xlim_);
ylim([0, max(pfun_sav(ixmax,:))]);
xticks(xticks_);
legend(labels, 'Location', 'NorthWest');

% Plot optimal consumption
subplot(1,2,2);
for iy = 1:par.N_y
    plot(par.grid_a, pfun_cons(:,iy), 'LineWidth', 1.5, 'Color', colours(:,iy));
    hold on;
end
hold off;
% Use custom function to set common plot style
set_ax_plot_style(gca, aspect);
title('Consumption');
xlabel('Assets');
% Set plot limits and ticks
xlim(xlim_);
ylim([min(pfun_cons(1,:)), max(pfun_cons(ixmax,:))]);
xticks(xticks_);

%% Export figure to PDF

% Directory where graphs should be stored
graphdir = '../graphs';
% Create directory if it does not exist
mkdir(graphdir)

% Custom function to remove excessive margins around subplots
papersize = [6.0 3.0];
layout = [1 2];
tightlayout(gcf, layout, 0.02, papersize);

% Store graphs as PDF, add number of asset grid points to file name.
fn = sprintf('EGM_labour_risk_N%d.pdf', par.N_a);
fn = fullfile(graphdir, fn);
saveas(gcf, fn)

