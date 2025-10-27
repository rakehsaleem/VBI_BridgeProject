function [Beam] = B03_Beam_Matrices(Beam)

% Generates the FEM system matrices for the beam model

% Copyright (c) 2019, Daniel Cantero. All rights reserved.
% Use, copy or distribution of this program is only allowed with the 
% explicit authorization from the author. Please contact daniel.cantero@ntnu.no

% -------------------------------------------------------------------------
% % ---- Inputs ----
% Beam = Structure with Beam's variables, including at least:
%   .Mesh.Node.a = Vector with elements size on X direction
%   .Prop.E_n = Beam Young's Modulus
%       Array of values giving the E for each element (same size as Beam.Mesh.Ele.a)
%   .Prop.I_n = Beam's section Second moment of Inertia product
%       Array of values giving the I for each element (same size as Beam.Mesh.Ele.a)
%   .Prop.rho_n = Beam's density
%       Array of values giving the rho for each element (same size as Beam.Mesh.Ele.a)
%   .Prop.A_n = Beam's section Area
%       Array of values giving the A for each element (same size as Beam.Mesh.Ele.a)
%   .BC.DOF_fixed = Array of DOF with fixed boundary condition
%   .BC.DOF_with_values = Array of DOF that have some additional stiffness
%   .BC.DOF_stiff_values = Array of stiffness values to be added to Beam.BC.DOF_with_values
%   .BC.num_DOF_fixed = Number of fixed DOF
%   .BC.num_DOF_with_values = Number of values with additional stiffness
%   .SysM.Opt.Mconsist_value = Option to select the type of Mass matrix
%       1 = Consitent mass matrix (Default)
%       2 = Lumped mass matrix
% % ---- Outputs ----
% Beam = Addition of fields to structure Beam:
%   .SysM.K = Global Stiffness matrix
%   .SysM.M = Global Mass matrix
% -------------------------------------------------------------------------

% Generate nodal connectivity for each elemnt
nodalconnec = [(1:Beam.Mesh.Ele.num);(1:Beam.Mesh.Ele.num)+1]';

% Initialize matrices
Beam.SysM.K = zeros(Beam.Mesh.DOF.num); 
CMM = Beam.SysM.K;

% Elemental matrices (In-line functions) (more efficient alternative to subfunctions)
B04_Beam_ele_M = @(r,A,L) double(r*A*L/420*[[156,22*L,54,-13*L];[22*L,4*L^2,13*L,-3*L^2];...
    [54,13*L,156,-22*L];[-13*L,-3*L^2,-22*L,4*L^2]]);
B05_Beam_ele_K = @(EI,L) double(EI/L^3*[[12,6*L,-12,6*L];[6*L,4*L^2,-6*L,2*L^2];...
    [-12,-6*L,12,-6*L];[6*L,2*L^2,-6*L,4*L^2]]);

% ---- Stiffness and Consistent Mass Matrices ----
for iel = 1:Beam.Mesh.Ele.num

    index = [(nodalconnec(iel,1)-1)*2+(1:2),(nodalconnec(iel,2)-1)*2+(1:2)];
    
    % Element matrices
    Me = B04_Beam_ele_M(Beam.Prop.rho_n(iel),Beam.Prop.A_n(iel),Beam.Mesh.Ele.a(iel));
    Ke = B05_Beam_ele_K(Beam.Prop.E_n(iel)*Beam.Prop.I_n(iel),Beam.Mesh.Ele.a(iel));
    
    % Assemblry of matrices
    Beam.SysM.K(index,index) = Beam.SysM.K(index,index) + Ke;
    CMM(index,index) = CMM(index,index) + Me;

end %for iel = 1:Beam.Mesh.Ele.num

if Beam.SysM.Opt.Mconsist_value ~= 1

    % ---- Lumped Mass Matrix (LMM) ----
    % Initialize matrix
    LMM = Beam.SysM.K*0;
    
    for iel = 1:Beam.Mesh.Ele.num

        index = [(nodalconnec(iel,1)-1)*2+(1:2),(nodalconnec(iel,2)-1)*2+(1:2)];
        
        % Element matrix
        Me = Beam.Prop.rho_n(iel)*Beam.Prop.A_n(iel)*Beam.Mesh.Ele.a(iel)*diag([1,0,1,0]);
        
        % LMM assembly
        LMM(index,index) = LMM(index,index) + Me;

    end %for iel = 1:Beam.Mesh.Ele.num

    % Mass Matrix
    Beam.SysM.M = Beam.SysM.Opt.Mconsist_value*CMM + (1-Beam.SysM.Opt.Mconsist_value)*LMM;
    
else % if Beam.SysM.Opt.Mconsist_value ~= 1

    Beam.SysM.M = CMM;
    
end % if Beam.SysM.Opt.Mconsist_value ~= 1

% ---- Application of boundary conditions ----
% Diagonal elements equal 1, and columns and rows equal zero for bc DOF

% Fixed DOF
Beam.SysM.K(Beam.BC.DOF_fixed,:) = 0; Beam.SysM.K(:,Beam.BC.DOF_fixed) = 0;
Beam.SysM.M(Beam.BC.DOF_fixed,:) = 0; Beam.SysM.M(:,Beam.BC.DOF_fixed) = 0;
for i = 1:Beam.BC.num_DOF_fixed
    Beam.SysM.K(Beam.BC.DOF_fixed(i),Beam.BC.DOF_fixed(i)) = Beam.BC.DOF_fixed_value;
    Beam.SysM.M(Beam.BC.DOF_fixed(i),Beam.BC.DOF_fixed(i)) = Beam.BC.DOF_fixed_value;
end % for i = 1:Beam.BC.num_DOF_fixed

% Support DOF with values
for i = 1:Beam.BC.num_DOF_with_values
    Beam.SysM.K(Beam.BC.DOF_with_values(i),Beam.BC.DOF_with_values(i)) = ...
        Beam.SysM.K(Beam.BC.DOF_with_values(i),Beam.BC.DOF_with_values(i)) + ...
        Beam.BC.DOF_stiff_values(i);
end % for i = 1:Beam.BC.num_DOF_values

% ---- Making output sparse ----
Beam.SysM.K = sparse(Beam.SysM.K); 
Beam.SysM.M = sparse(Beam.SysM.M);

% ---- Beam element Shape function ----
% Beam.Mesh.Ele.shape_fun = @(x,a) [(a+2*x).*(a-x).^2./a.^3; x.*(a-x).^2./a.^2; x.^2.*(3*a-2*x)./a.^3; -x.^2.*(a-x)./a.^2];
% Beam.Mesh.Ele.shape_fun_p = @(x,a) [-(6*x.*(a - x))./a.^3; 1 - (x.*(4*a - 3*x))./a.^2; (6*x.*(a - x))./a.^3; -(x.*(2*a - 3*x))./a.^2];
% Beam.Mesh.Ele.shape_fun_pp = @(x,a) [(12*x)./a.^3 - 6./a.^2; (6*x)./a.^2 - 4./a; 6./a.^2 - (12*x)./a.^3; (6*x)./a.^2 - 2./a];
% - Simply adding the double() command makes it faster -
Beam.Mesh.Ele.shape_fun = @(x,a) double([(a+2*x).*(a-x).^2./a.^3; x.*(a-x).^2./a.^2; x.^2.*(3*a-2*x)./a.^3; -x.^2.*(a-x)./a.^2]);
Beam.Mesh.Ele.shape_fun_p = @(x,a) double([-(6*x.*(a - x))./a.^3; 1 - (x.*(4*a - 3*x))./a.^2; (6*x.*(a - x))./a.^3; -(x.*(2*a - 3*x))./a.^2]);
Beam.Mesh.Ele.shape_fun_pp = @(x,a) double([(12*x)./a.^3 - 6./a.^2; (6*x)./a.^2 - 4./a; 6./a.^2 - (12*x)./a.^3; (6*x)./a.^2 - 2./a]);

% % Symbolical derivation
% syms a x
% shape = [(a+2*x).*(a-x).^2./a.^3; x.*(a-x).^2./a.^2; x.^2.*(3*a-2*x)./a.^3; -x.^2.*(a-x)./a.^2];
% shape_p = simple(diff(shape,'x'))
% shape_pp = simple(diff(shape_p,'x'))

% % Graphical check
%  a = 1;
%  x = linspace(0,a,100);
% figure;
% shape_fun = Beam.Mesh.Ele.shape_fun(x,a);
% shape_fun_p = Beam.Mesh.Ele.shape_fun_p(x,a);
% shape_fun_pp = Beam.Mesh.Ele.shape_fun_pp(x,a);
% for shape = 1:4
%     subplot(3,4,shape);
%         plot(x,shape_fun(shape,:))
%     subplot(3,4,4+shape);
%         plot(x,shape_fun_p(shape,:))
%     subplot(3,4,8+shape);
%         plot(x,shape_fun_pp(shape,:))    
% end % for shape = 1:4

% % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
% function Me = B04_Beam_ele_M(r,A,L)
% 
% % Generation of beam element Consistent Mass Matrix
% 
% % % ---------------------------------------------------------------
% % % ---- Input ----
% % r = Element density
% % A = Element Area
% % L = Length of Beam element
% % % ---- Output ----
% % Me = Element consistent Mass matrix
% % % ---------------------------------------------------------------
% 
% Me = r*A*L/420 * ...
%     [[   156,   22*L,    54,  -13*L];...
%     [  22*L,  4*L^2,  13*L, -3*L^2];...
%     [    54,   13*L,   156,  -22*L];...
%     [ -13*L, -3*L^2, -22*L,  4*L^2]];
% 
% % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
% 
% function Ke = B05_Beam_ele_K(EI,L)
% 
% % Generation of beam element Stiffness Matrix
% 
% % % ---------------------------------------------------------------
% % % ---- Input ----
% % EI = Element Young's Modulus and Second moment of Inertia product
% % L = Length of Beam element
% % % ---- Output ----
% % Ke = Element Stiffness matrix
% % % ---------------------------------------------------------------
% 
% Ke = EI/L^3 * ...
%     [[  12,   6*L,  -12,   6*L];...
%     [ 6*L, 4*L^2, -6*L, 2*L^2];...
%     [ -12,  -6*L,   12,  -6*L];...
%     [ 6*L, 2*L^2, -6*L, 4*L^2]];

% ---- End of function ----