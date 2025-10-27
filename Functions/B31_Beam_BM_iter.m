function [Sol] = B31_Beam_BM_iter(Calc,Beam,Sol)

% It is essentially the same function as B31_Beam_BM 
% but it should be used inside the iterative procedure and it is called
% only if the iteration criteria depends on BM values.

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
% ---- Outputs ----
% Sol = Addition of fields to structure Sol:
%   .BM.value_xt = Bending moment for each node in time [Beam.Mesh.Node.num x Calc.Solver.num_t_beam]
%   .BM.max = Maximum Bending moment (whole length)
%   .BM.COP = Critical observation point
%   .BM.pCOP = Critical observation point in percentage of beam length
%   .BM.min = Minimum Bending moment (whole length)
%   .BM.max05 = Maximum Bending moment at mid-span
% -------------------------------------------------------------------------

% Calculations performed only if iteration criteria depends on BM results
if Calc.Proc.Iter.criteria == 2

    [Sol] = B31_Beam_BM(Calc,Beam,Sol);

end % if Calc.Proc.Iter.criteria == 2

% ---- End of function ----