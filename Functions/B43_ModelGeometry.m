function [Calc,Veh] = B43_ModelGeometry(Calc,Veh,Beam)

% This function outputs some useful information about the model geometry 
% based on the provided values

% Copyright (c) 2019, Daniel Cantero. All rights reserved.
% Use, copy or distribution of this program is only allowed with the 
% explicit authorization from the author. Please contact daniel.cantero@ntnu.no

% -------------------------------------------------------------------------
% % ---- Inputs ----
% Calc = Structure with Calc's variables, including at least:
% Veh = Structure with vehicle's variables, including at least:
% Beam = Structure with beam's variables, including at least:
%   .Prop.Lb = Beam's length
% % ---- Outputs ----
% Calc = Structure with Calc variables and new fields:
%   .Profile.needed_x0 = Furthest to the left x coordinate needed for this event
%   .Profile.needed_x_end = Furthest to the right x coordinate needed for this event
%   .Profile.needed_L = Total length of profile needed for this event
%   .Solver.t_end = Total simulation time
% Veh(veh_num) = Additional field in the Veh structure
%   .Prop.wheelbase = Maximum distance between wheels of a vehicle
%   .Prop.num_wheels = Number of vehicle wheels (axles)
%   .Pos.prof_x0 = Start of profile for vehicle "veh_num"
%   .Pos.prof_x0 = End of profile for vehicle "veh_num"
%   .Pos.min_t_end = Minimum time requried to simulate the crossing of the vehicle
% -------------------------------------------------------------------------

% Initialize values
Calc.Profile.needed_x0 = 0;
Calc.Profile.needed_x_end = Beam.Prop.Lb;
Calc.Solver.t_end = 0;

% Calculations for each vehicle
for veh_num = 1:Veh(1).Event.num_veh

    % Vehicle Wheelbase and number of wheels
    Veh(veh_num).Prop.wheelbase = Veh(veh_num).Prop.ax_dist(end);
    Veh(veh_num).Prop.num_wheels = length(Veh(veh_num).Prop.ax_sp);
    
    % Profile start (x0) and end (x_end) for each vehicle
    Veh(veh_num).Pos.prof_x0 = -Veh(veh_num).Prop.wheelbase + Veh(veh_num).Pos.x0*(Veh(veh_num).Pos.x0<0);
    Veh(veh_num).Pos.prof_x_end = Beam.Prop.Lb + Veh(veh_num).Prop.wheelbase + ...
        (Veh(veh_num).Pos.x0-Beam.Prop.Lb)*(Veh(veh_num).Pos.x0>0);
    
    % Minimum time to simulate
    Veh(veh_num).Pos.min_t_end = (Veh(veh_num).Pos.prof_x_end-Veh(veh_num).Pos.prof_x0-Veh(veh_num).Prop.wheelbase)/abs(Veh(veh_num).Pos.vel);
    
    % Minimum and maximum values for x0 and x_end
    Calc.Profile.needed_x0 = min(Calc.Profile.needed_x0,Veh(veh_num).Pos.prof_x0);
    Calc.Profile.needed_x_end = max(Calc.Profile.needed_x_end,Veh(veh_num).Pos.prof_x_end);
    
    % Simulation time
    Calc.Solver.t_end = max(Calc.Solver.t_end,Veh(veh_num).Pos.min_t_end);
    
end % for veh_num = 1: Veh(1).Event.num_veh

% Addition of free vibration time
Calc.Solver.t_end = Calc.Solver.t_end + Calc.Opt.free_vib_seconds;

% Total length of profile
Calc.Profile.needed_L = Calc.Profile.needed_x_end - Calc.Profile.needed_x0;

% ---- End of function ----