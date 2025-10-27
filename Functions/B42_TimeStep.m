function [Calc] = B42_TimeStep(Calc,Veh,Beam)

% Calculates the appropriate time step length for the problem, considering:
%   1) Maximum vehicle frequency
%   2) Defined maximum bridge modes considered
%   3) Defined minimum steps per second
%   4) Profile maximum frequency (spatial frequency x Vehicle velocity)
%   5) User-defined maximum accurate frequency

% Copyright (c) 2019, Daniel Cantero. All rights reserved.
% Use, copy or distribution of this program is only allowed with the 
% explicit authorization from the author. Please contact daniel.cantero@ntnu.no

% -------------------------------------------------------------------------
% ---- Input ----
% Calc = Structre with calculation variables. It should include at least:
% ---- Output ----
% Calc = Additional fields in the structure:
% -------------------------------------------------------------------------

% Criteria 1: Maximum Vehicle frequency
max_frq = max(Veh(1).Modal.f)*2;
for veh_num = 2:Veh(1).Event.num_veh
    max_frq = max(max_frq,max(Veh(veh_num).Modal.f)*2);
end % for veh_num = 2:Veh(1).Event.num_veh

% Criteria 2: Maximum Beam frequency considered
max_frq = [max_frq,Beam.Modal.f(Calc.Solver.min_Beam_modes_considered)*2];

% Criteria 3: Defined minimum steps per second
max_frq = [max_frq,Calc.Solver.min_t_steps_per_second];

% Criteria 4: Maximum road profile frequency (Max. Spatial Frq x Max Veh. velocity)
if Calc.Profile.type == -1
    [aux1] = B39_Profile_Load(Calc);
    max_frq = [max_frq,aux1.Profile.Spatial_frq.max*Veh(1).Event.max_vel*2];
else
    max_frq = [max_frq,Calc.Profile.Spatial_frq.max*Veh(1).Event.max_vel*2];
end % if Calc.Profile.type == -1

% Citeria 5: User-defined maximum accurate frequency
max_frq = [max_frq,Calc.Solver.max_accurate_frq*2];

% Definition of steps per second
[Calc.Solver.t_steps_per_second,Calc.Solver.t_steps_criteria] = max(max_frq);

% Text for step definition criteria
switch Calc.Solver.t_steps_criteria
    case 1
        Calc.Solver.t_steps_criteria_text = 'Maximum vehicle frequency';
    case 2
        Calc.Solver.t_steps_criteria_text = 'Maximum bridge mode considered';
    case 3
        Calc.Solver.t_steps_criteria_text = 'Minimum steps per second';
    case 4
        Calc.Solver.t_steps_criteria_text = 'Profile maximum frequency (spatial frequency x Vehicle velocity)';
    case 5
        Calc.Solver.t_steps_criteria_text = 'User-defined maximum accurate frequency';
end % switch

% ---- End of function ----