function [MC] = myMC_counter_and_pause(MC)

% Updating counter depending on the num of runs or total calculation time
% Also runs "mypausecmd" to be able to pause a MC simulation during its execution

% -------------------------------------------------------------------------
% ---- Input ----
% MC = Structure with MonteCarlo variables, including at least:
%   .Calc.counter = Indicates previous run number
%   Define one of the following two:
%   .Opt.nums_runs = number of runs to perform in the MC analysis
%   .Opt.simulation_time = Duration of the simulation
% ---- Output ----
% MC = Addition of new fields to the structure MC
%   .Calc.counter = Updated counter value for the current run
% -------------------------------------------------------------------------

% Checking if continue simulation
if isfield(MC.Opt,'num_runs')
    if MC.Opt.num_runs == MC.Calc.counter
        MC.Calc.continue = 0;
    end % if MC.Opt.num_runs == MC.Calc.counter
elseif isfield(MC.Opt,'simulation_time')
    if etime(clock,MC.Calc.PCtime.start+MC.Calc.PCtime.end) > 0
        MC.Calc.continue = 0;
    end % if etime(clock,MC.Calc.PCtime.start+MC.Calc.PCtime.end) > 0
end % if isfield(MC.Opt,'num_runs')

% Update counter
if MC.Calc.continue == 1
    MC.Calc.counter = MC.Calc.counter + 1;
end % if MC.Calc.continue == 1

% Pause simulation
mypausecmd

% ---- End of script ----