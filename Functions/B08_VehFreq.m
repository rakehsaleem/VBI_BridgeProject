function [Calc,Veh] = B08_VehFreq(Calc,Veh)

% Calculates the vehicle(s) frequencies given the system matrices

% Copyright (c) 2019, Daniel Cantero. All rights reserved.
% Use, copy or distribution of this program is only allowed with the 
% explicit authorization from the author. Please contact daniel.cantero@ntnu.no

% -------------------------------------------------------------------------
% ---- Input ----
% Veh = Structure with Vehicle's variables, including at least:
%   .SysM.M = Vehicle global Mass matrix
%   .SysM.K = Vehicle global Stiffness matrix
% Calc = Structure with Calc's variables, including at least:
%   .Opt.veh_frq = If value = 1, vehicle's natural frequencies are calculated
% ---- Output ----
% Veh(veh_num) = Addition of fields to structure Veh:
%   Note: one substructure of each vehicle in the event
%   .Modal.w = Circular frequencies of vehicle
%   .Modal.f = Frequencies of vehicle
% -------------------------------------------------------------------------

% To obtain the natural frequencies, simply calculate the eigenvalues of
% the dynamic system. 
%   For M*xddot + C*xdot + K*xdot = F
% No damping and no external force is considered, then calculate:
%   eig(M\F)

if Calc.Opt.veh_frq == 1

    for veh_num = 1:Veh(1).Event.num_veh
    
        % ---- Eigenvalue analysis ----
        aux1 = eig(Veh(veh_num).SysM.K,Veh(veh_num).SysM.M);
        Veh(veh_num).Modal.w = sqrt(aux1);                      % Vehicle circuar frequencies (rad/s)
        Veh(veh_num).Modal.f = Veh(veh_num).Modal.w/(2*pi);     % Vehicle frequecies (Hz)
    
    end % for veh_num = 1:Veh(1).Event.num_veh

end % if Calc.Opt.veh_frq == 1

% ---- End of function ----