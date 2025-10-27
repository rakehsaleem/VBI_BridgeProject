function [Beam] = B22_DamageBeamElements(Beam)

% Changes the values of element properties according to the defined damage

% Copyright (c) 2019, Daniel Cantero. All rights reserved.
% Use, copy or distribution of this program is only allowed with the 
% explicit authorization from the author. Please contact daniel.cantero@ntnu.no

% -------------------------------------------------------------------------
% % ---- Inputs ----
% Beam = Structure with Beam's variables, including at least:
%   .Damage.type
%   .Damage.E / I / rho / A
% % ---- Outputs ----
% Beam = Change of fields of structure Beam:
%   .Prop.E_n = Beam Young's Modulus
%       Array of values giving the E for each element (same size as Beam.Mesh.Ele.a)
%   .Prop.I_n = Beam's section Second moment of Inertia product
%       Array of values giving the I for each element (same size as Beam.Mesh.Ele.a)
%   .Prop.rho_n = Beam's density
%       Array of values giving the rho for each element (same size as Beam.Mesh.Ele.a)
%   .Prop.A_n = Beam's section Area
%       Array of values giving the A for each element (same size as Beam.Mesh.Ele.a)
% -------------------------------------------------------------------------

% -- 1 element Damage --
if Beam.Damage.type == 1
    mid_ele_coord = Beam.Mesh.Ele.acum(1:end-1)+Beam.Mesh.Ele.a/2;
    if isfield(Beam.Damage,'E')
        [~,aux1] = min(abs(mid_ele_coord/Beam.Prop.Lb-Beam.Damage.E.per(1)/100))
        Beam.Prop.E_n(aux1) = Beam.Prop.E_n(aux1)*(1-Beam.Damage.E.per(2)/100);
    end % if isfield(Beam.Damage,'E')
    if isfield(Beam.Damage,'I')
        [~,aux1] = min(abs(mid_ele_coord/Beam.Prop.Lb-Beam.Damage.I.per(1)/100));
        Beam.Prop.I_n(aux1) = Beam.Prop.I_n(aux1)*(1-Beam.Damage.I.per(2)/100);
    end % if isfield(Beam.Damage,'I')
    if isfield(Beam.Damage,'rho')
        [~,aux1] = min(abs(mid_ele_coord/Beam.Prop.Lb-Beam.Damage.rho.per(1)/100));
        Beam.Prop.rho_n(aux1) = Beam.Prop.rho_n(aux1)*(1-Beam.Damage.rho.per(2)/100);
    end % if isfield(Beam.Damage,'rho')
    if isfield(Beam.Damage,'A')
        [~,aux1] = min(abs(mid_ele_coord/Beam.Prop.Lb-Beam.Damage.A.per(1)/100));
        Beam.Prop.A_n(aux1) = Beam.Prop.A_n(aux1)*(1-Beam.Damage.A.per(2)/100);
    end % if isfield(Beam.Damage,'A')
    %Edited by Zohaib for Neutral Axis Change
    if isfield(Beam.Damage,'h')
        [~,aux1] = min(abs(mid_ele_coord/Beam.Prop.Lb-Beam.Damage.h.per(1)/100));
        Beam.Prop.h_n(aux1) = Beam.Prop.h_n(aux1)*(1-Beam.Damage.h.per(2)/100);
    end 
% -- Global Damage --
elseif Beam.Damage.type == 2    
    if isfield(Beam.Damage,'E')
        Beam.Prop.E_n = Beam.Prop.E_n*(1-Beam.Damage.E.per(2)/100);
    end % if isfield(Beam.Damage,'E')
    if isfield(Beam.Damage,'I')
        Beam.Prop.I_n = Beam.Prop.I_n*(1-Beam.Damage.I.per(2)/100);
    end % if isfield(Beam.Damage,'I')
    if isfield(Beam.Damage,'rho')
        Beam.Prop.rho_n = Beam.Prop.rho_n*(1-Beam.Damage.rho.per(2)/100);
    end % if isfield(Beam.Damage,'rho')
    if isfield(Beam.Damage,'A')
        Beam.Prop.A_n = Beam.Prop.A_n*(1-Beam.Damage.A.per(2)/100);
    end % if isfield(Beam.Damage,'A')
%         %Edited by Zohaib for Neutral Axis Change
%     if isfield(Beam.Damage,'A')
%         Beam.Prop.h_n = Beam.Prop.h_n*(1-Beam.Damage.h.per(2)/100);
%     end
end % if Beam.Damage.ype == 1

% ---- End of script ----