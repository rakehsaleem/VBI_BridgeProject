function [Sol] = B33_Beam_Shear(Calc,Beam,Sol,varargin)

% Calculates the Shear of the beam using the nodal displacements

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
%   .Opt.calc_mode_V = To select the mode of V calculation
%       0 = Calculations done node by node
%       1 = Average results at the node 
% ---- Optional inputs ----
% static_flag = Calculate the static shear if this optional input is equal to 1
% ---- Outputs ----
% Sol = Addition of fields to structure Sol:
%   .Shear.value_xt = Shear for each node in time [Beam.Nodes.Tnum x Calc.Time.nt_beam]
%   .Shear.Max.value = Maximum shear value
%   .Shear.Max.COP = Critical observation point of Maximum
%   .Shear.Max.pCOP = Critical observation point in percentage of beam length of Maximum
%   .Shear.Max.cri_t_ind = Critical time index
%   .Shear.Min.value = Minimum shear value
%   .Shear.Min.value05 = Minimium shear value at mid-span
%   .Shear.Min.COP = Critical observation point of Minimum
%   .Shear.Min.pCOP = Critical observation point in percentage of beam length of Minimum
%   .Shear.Min.cri_t_ind = Critical time index
% -------------------------------------------------------------------------

% Input processing
static_flag = 0;
if nargin > 3
    static_flag = varargin{1};
end % if nargin > 3

% Field name
if static_flag == 1
    Shear_field_name = 'Shear_static';
    U_field_name = 'U_static';
else
    Shear_field_name = 'Shear';
    U_field_name = 'U';
end % if static_flag == 1
U_subfield_name = 'value_DOFt';

% Initialize variables
Sol.Beam.(Shear_field_name).value_xt = zeros(Beam.Mesh.Node.num,Calc.Solver.num_t_beam);

% ---- NO average nodal values ----
if Calc.Opt.calc_mode_Shear == 0
    
    for ix = 1:Beam.Mesh.Node.num-1

        aux1 = B32_Beam_ele_HS(Beam.Mesh.Ele.a(ix),Beam.Prop.E_n(ix),Beam.Prop.I_n(ix));
        Sol.Beam.(Shear_field_name).value_xt(ix,:) = aux1(1,:) * Sol.Beam.(U_field_name).(U_subfield_name)((ix*2-1):(ix*2-1)+3,:);

    end % for ix

    ix = Beam.Mesh.Node.num;
    aux1 = B32_Beam_ele_HS(Beam.Mesh.Ele.a(ix-1),Beam.Prop.E_n(ix-1),Beam.Prop.I_n(ix-1));
    Sol.Beam.(Shear_field_name).value_xt(ix,:) = aux1(2,:) * Sol.Beam.(U_field_name).(U_subfield_name)(((ix-1)*2-1):((ix-1)*2-1)+3,:);

% ---- AVERAGE nodal values ----
elseif Calc.Opt.calc_mode_Shear == 1
    
    for ix = 1:Beam.Mesh.Node.num-1

        Sol.Beam.(Shear_field_name).value_xt([1,2]+(ix-1),:) = Sol.Beam.(Shear_field_name).value_xt([1,2]+(ix-1),:) + ...
            B32_Beam_ele_HS(Beam.Mesh.Ele.a(ix),Beam.Prop.E_n(ix),Beam.Prop.I_n(ix)) * ... 
            Sol.Beam.(U_field_name).(U_subfield_name)((ix*2-1):(ix*2-1)+3,:);

    end % for ix

    % Average of nodes with multiple calculations
    Sol.Beam.(Shear_field_name).value_xt(2:end-1,:) = Sol.Beam.(Shear_field_name).value_xt(2:end-1,:)/2;
    
end % Calc.Options.calc_mode_Shear

% ---- End of function ----