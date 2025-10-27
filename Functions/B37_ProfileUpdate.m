function [Calc,Sol] = B37_ProfileUpdate(Calc,Veh,Beam,Sol)

% In the first iteration:
%   Initializes necessary variables
% Subsequent iterations:
%   Updates the profile with the beam deformation

% Copyright (c) 2019, Daniel Cantero. All rights reserved.
% Use, copy or distribution of this program is only allowed with the 
% explicit authorization from the author. Please contact daniel.cantero@ntnu.no

% -------------------------------------------------------------------------
% ---- Input ----
% Calc = Structure with Calculation variables, including at least:
%    .num_iter = Iteration number
%    .Solver.num_t_beam = Number of time steps for beam calculations
%    .h_path = 
%    .t0_ind_beam = 
%    .t_end_ind_beam = 
%    .dt = 
% Veh = Structure with Vehicle variables, including at least:
%    .num_wheels = 
% Beam = Structure with Beam variables, including at least:
%    .Mesh.Node.num
% ---- Output ----
% Calc = Addition of fields to structure Calc:
%    .def_under = Beam deformation under vehicle wheels [Veh.Prop.num_wheels,Calc.Solver.num_t_beam]
%    .vel_under = Beam "velocity" under vehicle wheels [Veh.Prop.num_wheels,Calc.Solver.num_t_beam]
%    .old_BM_value_xt = Dummy variable to keep previous BM results [Beam.Mesh.Node.num,Calc.Solver.num_t_beam]
%    .def_under_old = Dummy variable to keep previous def_under results
%    .h_def = 
%    .hd_def = 
% -------------------------------------------------------------------------

if Calc.Proc.Iter.num == 0
    
    % Initialize variables
    for veh_num = 1:Veh(1).Event.num_veh
        Sol.Veh(veh_num).Under.def = zeros(Veh(veh_num).Prop.num_wheels,Calc.Solver.num_t_beam);
        Sol.Veh(veh_num).Under.def_old = Sol.Veh(veh_num).Under.def + 1;
        Sol.Veh(veh_num).Under.vel = zeros(Veh(veh_num).Prop.num_wheels,Calc.Solver.num_t_beam);
    end % for veh_num = 1:Veh(1).Event.num_veh
    Calc.Aux.old_BM_value_xt = zeros(Beam.Mesh.Node.num,Calc.Solver.num_t_beam);
    
end % if Calc.Proc.Iter.num == 0
    
% -- Addition of previous deformation to original profile --
for veh_num = 1:Veh(1).Event.num_veh
    Sol.Veh(veh_num).Under.h = Veh(veh_num).Pos.wheels_h(:,Calc.Solver.t0_ind_beam:Calc.Solver.t_end_ind_beam) + ...
        + Sol.Veh(veh_num).Under.def;
    Sol.Veh(veh_num).Under.hd = Veh(veh_num).Pos.wheels_hd(:,Calc.Solver.t0_ind_beam:Calc.Solver.t_end_ind_beam) + ...
        + Sol.Veh(veh_num).Under.vel;
end % for veh_num = 1:Veh(1).Event.num_veh

% Iterations counter
Calc.Proc.Iter.num = Calc.Proc.Iter.num + 1;

% Graphical check
% figure; hold on; box on;
%     for wheel_num = 1:Veh(1).Prop.num_wheels
%         plot(Veh(1).Pos.wheels_x(wheel_num,Veh(1).Pos.t0_ind_beam:Veh(1).Pos.t_end_ind_beam),Sol.Veh(1).Under.h(wheel_num,:));
%     end % for wheel_num = 1:Veh(1).Prop.num_wheels
%     title(['h for iteration ',num2str(Calc.Proc.Iter.num)]);
%     drawnow;
%     pause(1);

% ---- End of function ----