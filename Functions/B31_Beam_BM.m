function [Sol] = B31_Beam_BM(Calc,Beam,Sol,varargin)

% Calculates the BM of the beam using the nodal displacements

% Copyright (c) 2019, Daniel Cantero. All rights reserved.
% Use, copy or distribution of this program is only allowed with the 
% explicit authorization from the author. Please contact daniel.cantero@ntnu.no

% -------------------------------------------------------------------------
% % ---- Inputs ----
% Sol = Structure with beam Solutions's variables.
%   .Beam.U.value_DOFt = Nodal displacements of the beam FEM
% Beam = Structure with Beam's variables, including at least:
%   .Mesh.Node.num = Total number of nodes
%   .a = Vector with elements size on X direction
%   .E_n = Vector with elements Young's modulus
%   .I_n = Vector with elements Moments of Inertia
%   .Mesh.Node.at_mid_span = Mid-span node index
%   .Lb = Beam length
%   .acum = Cumulative node distances array
% Calc = Structure with Calculation variables, including at least:
%   .Solver.num_t_beam = Number of time steps for beam calculations
%   .Opt.calc_mode_BM = To select the mode of BM calculation
%       0 = Calculations done node by node
%       1 = Average results at the node 
% ---- Optional inputs ----
% static_flag = Calculate the static BM if this optional input is equal to 1
% ---- Outputs ----
% Sol = Addition of fields to structure Sol:
%   .BM.value_xt = Bending moment for each node in time [Beam.Mesh.Node.num x Calc.Solver.num_t_beam]
%   .BM.Max.value = Maximum Bending moment (whole length)
%   .BM.Max.value05 = Maximum Bending moment at mid-span
%   .BM.Max.COP = Critical observation point
%   .BM.Max.pCOP = Critical observation point in percentage of beam length
%   .BM.Max.cri_t_ind = Critical time index
%   .BM.Min.value = Minimum Bending moment (whole length)
%   .BM.Min.value05 = Minimum Bending moment at mid-span
%   .BM.Min.COP = Critical observation point
%   .BM.Min.pCOP = Critical observation point in percentage of beam length
%   .BM.Min.cri_t_ind = Critical time index
% -------------------------------------------------------------------------

% Input processing
static_flag = 0;
if nargin > 3
    static_flag = varargin{1};
end % if nargin > 3

% Field name
if static_flag == 1
    BM_field_name = 'BM_static';
    U_field_name = 'U_static';
else
    BM_field_name = 'BM';
    U_field_name = 'U';
end % if static_flag == 1
U_subfield_name = 'value_DOFt';

% Initialize variables
Sol.Beam.(BM_field_name).value_xt = zeros(Beam.Mesh.Node.num,Calc.Solver.num_t_beam);

% ---- NO average nodal values ----
if Calc.Opt.calc_mode_BM == 0
    
    for ix = 1:Beam.Mesh.Node.num-1

        aux1 = B30_Beam_ele_H(Beam.Mesh.Ele.a(ix),Beam.Prop.E_n(ix),Beam.Prop.I_n(ix));
        Sol.Beam.(BM_field_name).value_xt(ix,:) = aux1(1,:) * Sol.Beam.(U_field_name).(U_subfield_name)((ix*2-1):(ix*2-1)+3,:);

    end % for ix

    ix = Beam.Mesh.Node.num;
    aux1 = B30_Beam_ele_H(Beam.Mesh.Ele.a(ix-1),Beam.Prop.E_n(ix-1),Beam.Prop.I_n(ix-1));
    Sol.Beam.(BM_field_name).value_xt(ix,:) = aux1(2,:) * Sol.Beam.(U_field_name).(U_subfield_name)(((ix-1)*2-1):((ix-1)*2-1)+3,:);
    
% ---- AVERAGE nodal values ----
elseif Calc.Opt.calc_mode_BM == 1
    
    for ix = 1:Beam.Mesh.Node.num-1

        Sol.Beam.(BM_field_name).value_xt([1,2]+(ix-1),:) = Sol.Beam.(BM_field_name).value_xt([1,2]+(ix-1),:) + ...
            B30_Beam_ele_H(Beam.Mesh.Ele.a(ix),Beam.Prop.E_n(ix),Beam.Prop.I_n(ix)) * ... 
            Sol.Beam.(U_field_name).(U_subfield_name)((ix*2-1):(ix*2-1)+3,:);

    end % for ix

    % Average of nodes with multiple calculations
    Sol.Beam.(BM_field_name).value_xt(2:end-1,:) = Sol.Beam.(BM_field_name).value_xt(2:end-1,:)/2;
    
end % Calc.Opt.calc_mode_BM

% ---- End of function ----