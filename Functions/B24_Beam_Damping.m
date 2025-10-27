function [Beam] = B24_Beam_Damping(Beam)

% Calculates the Beam damping matrix 
%   Rayleigh damping is addopted 
%   1st and 2nd beam frequencies are taken as reference (of non-rigid modes)

% Copyright (c) 2019, Daniel Cantero. All rights reserved.
% Use, copy or distribution of this program is only allowed with the 
% explicit authorization from the author. Please contact daniel.cantero@ntnu.no

% -------------------------------------------------------------------------
% % ---- Inputs ----
% Beam = Structure with Beam's variables, including at least:
%   .Mesh.Ele.a = Vector with elements size on X direction
%   .Prop.E_n = Beam Young's Modulus
%       Array of values giving the E for each element (same size as Beam.Mesh.Ele.a)
%   .Prop.I_n = Beam's section Second moment of Inertia product
%       Array of values giving the I for each element (same size as Beam.Mesh.Ele.a)
%   .Prop.rho_n = Beam's density
%       Array of values giving the rho for each element (same size as Beam.Mesh.Ele.a)
%   .Prop.A_n = Beam's section Area
%       Array of values giving the A for each element (same size as Beam.Mesh.Ele.a)
%   .SysM.Opt.Mconsist_value = Option to select the type of Mass matrix
%       1 = Consitent mass matrix (Default)
%       2 = Lumped mass matrix
% % ---- Outputs ----
% Beam = Addition of fields to structure Beam:
%   .SysM.K = Global Stiffness matrix
%   .SysM.M = Global Mass matrix
% -------------------------------------------------------------------------

if Beam.Prop.damp_xi > 0
    
    % Reference frequencies
    ref_w = Beam.Modal.w((1:2)+Beam.Modal.num_rigid_modes);
    
    % Rayleigh's coefficients 'alpha' and 'beta'
    aux1 = (1/2)*[[1/ref_w(1) ref_w(1)];[1/ref_w(2) ref_w(2)]]\[Beam.Prop.damp_xi;Beam.Prop.damp_xi];

    % Damping matrix
    Beam.SysM.C = aux1(1)*Beam.SysM.M + aux1(2)*Beam.SysM.K;

else
    
    % No Damping case
    Beam.SysM.C = Beam.SysM.K*0;
    
end % if Beam.Prop.damp_xi > 0

% ---- End of script ----