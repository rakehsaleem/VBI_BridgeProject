function [Sol] = B15_Beam_Static_U(Calc,Veh,Beam,Sol)

% Calculates the static deformation of the Beam due to static load of the vehicle(s)

% Copyright (c) 2019, Daniel Cantero. All rights reserved.
% Use, copy or distribution of this program is only allowed with the 
% explicit authorization from the author. Please contact daniel.cantero@ntnu.no

% -------------------------------------------------------------------------
% ---- Input ----
% Beam = Structure with Beam's variables, including at least:
%   .SysM.K = Global Stiffness matrix
%   .Mesh.DOF.num = Total number of DOF
%   .acum = X coordinate of each node
%   .Mesh.Ele.nodes = Based on ele_nodes. Each row includes the DOF asociated to every element.
%       Each element represents a row.
%   .bc = Vector containing DOF with boundary conditions
% Calc = Structure with Calculation variables, including at least:
%   .tn = Total number of time steps
%   .grav = Gravity
%   .tn = Total number of time steps
%   .F = External force for every time step
%   .x = Location of external for for every time step
% Veh = Addition of fields to structure Veh:
%   .m1 = Vehicle mass
% ---- Output ----
% Sol.Beam.U_static.value_DOFt = Vertical displacement due to the static load of vehicle
% -------------------------------------------------------------------------

% Vehicle loop
for veh_num = 1:Veh(1).Event.num_veh

    % Temporary copy of vehicle force on beam
    Aux.Veh(veh_num).Under.onBeamF = Sol.Veh(veh_num).Under.onBeamF;
    % Definition of static force in time
    Sol.Veh(veh_num).Under.onBeamF = Veh(veh_num).Static.load*ones(1,Calc.Solver.num_t_beam);
    
end % for veh_num = 1:Veh(1).Event.num_veh

% Nodal forces calculation
[F] = B14_EqVertNodalForce(Calc,Veh,Beam,Sol);

% Static beam deformation
Sol.Beam.U_static.value_DOFt = Beam.SysM.K\F;

% Restoring original values of force on beam
for veh_num = 1:Veh(1).Event.num_veh
    Sol.Veh(veh_num).Under.onBeamF = Aux.Veh(veh_num).Under.onBeamF;
end % for veh_num = 1:Veh(1).Event.num_veh

% ---- End of function ----