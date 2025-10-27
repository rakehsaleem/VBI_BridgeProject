function [Sol] = B47_LoadEffectsMax(Beam,Sol)

% To extract maximum/minimum of all the calculated beam load effects

% Copyright (c) 2019, Daniel Cantero. All rights reserved.
% Use, copy or distribution of this program is only allowed with the 
% explicit authorization from the author. Please contact daniel.cantero@ntnu.no

% -------------------------------------------------------------------------
% % ---- Inputs ----
% Sol = Structure with solution information, with:
%   .Beam.(LoadEffect).value_xt = Matrix with values of load effect in space and time (for BM and Shear)
%   alternativelly
%   .Beam.(LoadEffect).value_DOFt = Matrix with values of load effect for each DOF and time (for U, V, A)
% Beam = Structure with Beam's variables, including at least:
%   .Prop.Lb = Beam length
%   .Mesh.acum = Cumulative node distances array
%   .Mesh.mid_span_node_ind = Mid-span node index
% ---- Outputs ----
% Sol = Addition of fields to structure:
%   .Beam.(LoadEffect).value_xt = BM for each node in time [Beam.Nodes.Tnum x Calc.Time.nt_beam]
%   .Beam.(LoadEffect).Max.value = Maximum value
%   .Beam.(LoadEffect).Max.value05 = Maximum vertical at mid-span
%   .Beam.(LoadEffect).Max.COP = Critical observation point
%   .Beam.(LoadEffect).Max.pCOP = Critical observation point in percentage of beam length
%   .Beam.(LoadEffect).Min.value = Minimum value
%   .Beam.(LoadEffect).Min.value05 = Minimum vertical at mid-span
%   .Beam.(LoadEffect).Min.COP = Critical observation point
%   .Beam.(LoadEffect).Min.pCOP = Critical observation point in percentage of beam length
% -------------------------------------------------------------------------

% Load effects to consider
field_names = fields(Sol.Beam);

% Load effect loop
for field_num = 1:length(field_names)
    
    % Field name
    field_name = field_names{field_num};

    % Create value_xt variable (if necessary)
    if isfield(Sol.Beam.(field_name),'value_DOFt')
        Sol.Beam.(field_name).value_xt = Sol.Beam.(field_name).value_DOFt(1:2:end,:);
    end % if isfield(Sol.Beam.(field_name),'value_DOFt')
    
    % Maximum Bending Moment
    [Sol.Beam.(field_name).Max.value,aux1] = max(Sol.Beam.(field_name).value_xt);
    [Sol.Beam.(field_name).Max.value,aux2] = max(Sol.Beam.(field_name).Max.value);
    Sol.Beam.(field_name).Max.COP = Beam.Mesh.Ele.acum(aux1(aux2));
    Sol.Beam.(field_name).Max.pCOP = Sol.Beam.(field_name).Max.COP/Beam.Prop.Lb*100;
    Sol.Beam.(field_name).Max.cri_t_ind = aux2;

    % Mid-span Maximum BM
    Sol.Beam.(field_name).Max.value05 = max(Sol.Beam.(field_name).value_xt(Beam.Mesh.Node.at_mid_span,:));

    % Minimum Bending Moment
    [Sol.Beam.(field_name).Min.value,aux1] = min(Sol.Beam.(field_name).value_xt);
    [Sol.Beam.(field_name).Min.value,aux2] = min(Sol.Beam.(field_name).Min.value);
    Sol.Beam.(field_name).Min.COP = Beam.Mesh.Ele.acum(aux1(aux2));
    Sol.Beam.(field_name).Min.pCOP = Sol.Beam.(field_name).Min.COP/Beam.Prop.Lb*100;
    Sol.Beam.(field_name).Min.cri_t_ind = aux2;

    % Mid-span Bending Moment
    Sol.Beam.(field_name).Min.value05 = min(Sol.Beam.(field_name).value_xt(Beam.Mesh.Node.at_mid_span,:));

end % for field_num = 1:length(field_names)

% ---- End of function ----