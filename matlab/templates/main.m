%
% Topics in Macroeconomics (ECON5098), 2021-22
%
% Template main file that can be used as a starting point to implement
% household problem solvers.
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

%% Grids

% Asset grid: allocate more points towards the left end, i.e., at lower 
% asset levels.
% Step 1: create grid on [0,1] that is more dense for smaller values
grid_01 = linspace(0.0, 1.0, par.N_a) .^ 1.3;
% Step 2: Rescale onto desired asset range
grid_a = par.a_min + (par.a_max - par.a_min) * grid_01;

% Store asset grid as column vector!
par.grid_a = grid_a';

%% Run household problem solver (VFI, EGM)

% Termination tolerance
tol = 1.0e-6;
% Max. number of iterations
maxiter = 1000;

% The solver should return the value function (if applicable) and the
% savings policy function, pfun_sav.

% INSERT CALL TO SOLVER HERE


%% Recover consumption policy function

% Cash-at-hand at beginning of period
cah = (1.0 + par.r) * par.grid_a + par.y;

% Consumption policy function (optimal consumption level)
pfun_cons = cah - pfun_sav;

%% Plot value and policy functions (simple plotting)

% Plot value functions (if applicable, i.e. only for VFI)
subplot(1,3,1);
plot(par.grid_a, vfun);
title('Value function');
xlabel('Assets');

% Plot savings (i.e. next-period assets)
subplot(1,3,2);
plot(par.grid_a, pfun_sav);
title('Savings');
xlabel('Assets');

% Plot optimal consumption
subplot(1,3,3);
plot(par.grid_a, pfun_cons);
title('Consumption');
xlabel('Assets');
