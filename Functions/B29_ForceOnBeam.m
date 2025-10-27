function [Sol] = B29_ForceOnBeam(Calc,Veh,Sol)

% Calculates the force on the beam due to the vehicle(s)

% Copyright (c) 2019, Daniel Cantero. All rights reserved.
% Use, copy or distribution of this program is only allowed with the 
% explicit authorization from the author. Please contact daniel.cantero@ntnu.no

% -------------------------------------------------------------------------
% ---- Input ----
% Calc = Structure with Calculation variables, including at least:
%   .Solver.num_t_beam
% Veh = Structure with Vehicle's variables, including at least:
%   .ktn
%   .ctn
%   .sta_loads
% Sol = Structure with Solution's variables, including at least:
%   .Veh.Wheels.Urel = Wheels' relative displacements
%   .Veh.Wheels.Vrel = Wheels' relative velocities
% ---- Output ----
% Calc = Additional fields in the structure:
%   .onBeamF = Array containing the vertical forces to be applied on the
%       beam. The dimensions of the array are:
%       [Number of wheels x Calc.Solver.num_t_beam]
% -------------------------------------------------------------------------

for veh_num = 1:Veh(1).Event.num_veh
    
    % Relevant time indices
    inds = Calc.Solver.t0_ind_beam:Calc.Solver.t_end_ind_beam;
    
    % Due to relative wheel displacements (and velocities)
    Sol.Veh(veh_num).Under.onBeamF = ...
        Sol.Veh(veh_num).Wheels.Urel(:,inds).*(Veh(veh_num).Prop.kTi'*ones(1,Calc.Solver.num_t_beam)) + ...
        Sol.Veh(veh_num).Wheels.Vrel(:,inds).*(Veh(veh_num).Prop.cTi'*ones(1,Calc.Solver.num_t_beam));

    % Due to static vehicle load
    Sol.Veh(veh_num).Under.onBeamF = Sol.Veh(veh_num).Under.onBeamF + Veh(veh_num).Static.load*ones(1,Calc.Solver.num_t_beam);

end % for veh_num = 1:Veh.Event.num_veh

% ---- End of function ----