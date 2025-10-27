function [Sol] = B13_Beam_DynamicCalc(Calc,Veh,Beam,Sol)

% Solving the dynamic problem of a FEM beam due to a moving load

% Copyright (c) 2019, Daniel Cantero. All rights reserved.
% Use, copy or distribution of this program is only allowed with the 
% explicit authorization from the author. Please contact daniel.cantero@ntnu.no

% -------------------------------------------------------------------------
% ---- Input ----
% Beam = Structure with Beam's variables, including at least:
%   .SysM.M = Global Mass matrix
%   .SysM.C = Global Mass matrix (optional)
%   .SysM.K = Global Stiffness matrix
%   .Mesh.DOF.num = Total number of DOF
% Calc = Structure with Calculation variables, including at least:
%   .NewMark_damp = Selection of NewMark-Beta scheme
%       0 = No numerical damping
%       1 = With numerical damping
%   .dt = Time step
%   .tn = Total number of time steps
%   .F = External force for every time step
%   .x = Location of external for for every time step
% ---- Output ----
% Sol = Addition of fields to structure Sol:
%   .U = Results for vertical displacement
%   .V = Results for vertical velocities (1st derivative of displacement)
%   .A = Results for vertical acceleration (2nd derivative of displacement)
% Calc = Additional fields in the structure:
% -------------------------------------------------------------------------

% -- Effective Stiffness Matrix --
eff_K_beam = Beam.SysM.K + Beam.SysM.M/(Calc.Solver.NewMark.beta*Calc.Solver.dt^2) + ...
    Calc.Solver.NewMark.delta/(Calc.Solver.NewMark.beta*Calc.Solver.dt)*Beam.SysM.C;

% -- Initialize variables --
Sol.Beam.U.value_DOFt = zeros(Beam.Mesh.DOF.num,Calc.Solver.num_t_beam);
Sol.Beam.A.value_DOFt = Sol.Beam.U.value_DOFt; 
Sol.Beam.V.value_DOFt = Sol.Beam.A.value_DOFt;

% -- Force Matrix --
[F] = B14_EqVertNodalForce(Calc,Veh,Beam,Sol);

% -- Initial Static Calculation --
if isfield(Calc.Opt,'beamInitSta')
    Sol.Beam.U.value_DOFt(:,1) = Beam.SysM.K\F(:,1);
end % if isfield(Calc.Opt,'beamInitSta')

% -- Step by step calculation --
for t = 1:Calc.Solver.num_t_beam-1

    % ---- Beam System ----
    % Newmark-beta scheme (As seen in B014)
    % Note A ~= Sol.Beam.A; A = auxiliary variable; Sol.Beam.A = Beam Accelerations
    A = Sol.Beam.U.value_DOFt(:,t)/(Calc.Solver.NewMark.beta*Calc.Solver.dt^2) + ...
            Sol.Beam.V.value_DOFt(:,t)/(Calc.Solver.NewMark.beta*Calc.Solver.dt) + ...
            Sol.Beam.A.value_DOFt(:,t)*(1/(2*Calc.Solver.NewMark.beta)-1);
    B = (Calc.Solver.NewMark.delta/(Calc.Solver.NewMark.beta*Calc.Solver.dt)*Sol.Beam.U.value_DOFt(:,t) - ...
            (1-Calc.Solver.NewMark.delta/Calc.Solver.NewMark.beta)*Sol.Beam.V.value_DOFt(:,t) - ...
            (1-Calc.Solver.NewMark.delta/(2*Calc.Solver.NewMark.beta))*Calc.Solver.dt*Sol.Beam.A.value_DOFt(:,t));
    Sol.Beam.U.value_DOFt(:,t+1) = eff_K_beam\(F(:,t+1) + Beam.SysM.M*A + Beam.SysM.C*B);
    Sol.Beam.V.value_DOFt(:,t+1) = Calc.Solver.NewMark.delta/(Calc.Solver.NewMark.beta*Calc.Solver.dt)*Sol.Beam.U.value_DOFt(:,t+1) - B;
    Sol.Beam.A.value_DOFt(:,t+1) = Sol.Beam.U.value_DOFt(:,t+1)/(Calc.Solver.NewMark.beta*Calc.Solver.dt^2) - A;

end % for t

% ---- End of script ----