function [Sol] = B28_Veh_DynamicCalc_Approach(Calc,Veh,varargin)

% 1) Initializes vehicle(s) solution variables
% 2) Performs initial static calculation of vehicle (optional)
% 3) Solves the vehicle(s) dynamic response when it is traversing the approach

% Copyright (c) 2019, Daniel Cantero. All rights reserved.
% Use, copy or distribution of this program is only allowed with the 
% explicit authorization from the author. Please contact daniel.cantero@ntnu.no

% -------------------------------------------------------------------------
% ---- Input ----
% Calc = Structure with Calculation variables, including at least:
%   .NewMark_damp = Selection of NewMark-Beta scheme
%       0 = No numerical damping
%       1 = With numerical damping
%   .dt = Time step
%   .tn = Total number of time steps
%   .h_def = Profile for each time step (Include beam deformation in VBI models)
%   .hd_def = Profile for each time step (Include beam deformation in VBI models)
% Veh = Structure with Vehicle's variables, including at least:
%   .M = Vehicle global Mass matrix
%   .C = Vehicle global Damping matrix
%   .K = Vehicle global Stiffness matrix
% -- Optional inputs --
% varargin{1} = 0 = Only vehicle passing on the approach is calculated (default)
%             = 1 = then calculation also includes the vehicle crossing
%                   beam (and free vibrations). It calculates all time steps
% ---- Output ----
% Sol = Addition of fields to structure Sol:
%   .U = Results for vertical displacement
%   .Urel = Results for relative vertical displacement
%   .V = Results for vertical velocities
%   .Vrel = Results for relative vertical velocities
% Calc = Additional fields in the structure:
% -------------------------------------------------------------------------

for veh_num = 1:Veh(1).Event.num_veh

    % -- Initialize variable --
    Sol.Veh(veh_num).U = zeros(Veh(veh_num).DOF(1).num_independent,Calc.Solver.num_t);
    Sol.Veh(veh_num).A = Sol.Veh(veh_num).U;
    Sol.Veh(veh_num).V = Sol.Veh(veh_num).U;
    Sol.Veh(veh_num).Wheels.U = zeros(Veh(veh_num).Prop.num_wheels,Calc.Solver.num_t);
    Sol.Veh(veh_num).Wheels.Urel = Sol.Veh(veh_num).Wheels.U;
    Sol.Veh(veh_num).Wheels.V = Sol.Veh(veh_num).Wheels.U;
    Sol.Veh(veh_num).Wheels.Vrel = Sol.Veh(veh_num).Wheels.U;

    if Veh(veh_num).Pos.t0_ind_beam > 1

        num_t_app = Veh(veh_num).Pos.t0_ind_beam;
        if nargin == 3
            if varargin{1} == 1
                num_t_app = Calc.Solver.num_t-1;
            end % if varargin{1} == 1
        end % if nargin == 3

        % -- Initial Static Calculation -- (When vehicle only on the approach)
        if myIsfield(Calc.Opt,{'vehInitSta'},1)
            aux1 = Veh(veh_num).SysM.N2w'*(Veh(veh_num).Prop.kTi'.*Veh(veh_num).Pos.wheels_h(:,1));
            Sol.Veh(veh_num).U(:,1) = Veh(veh_num).SysM.K\aux1;
        end % if myIsfield(Calc.Opt,{'vehInitSta'},1)

        % ---- Dynamic Calculation on Approach ----

        % -- Effective stiffness matrix --
        eff_K_veh = Veh(veh_num).SysM.K + Veh(veh_num).SysM.M/(Calc.Solver.NewMark.beta*Calc.Solver.dt^2) + ...
            Calc.Solver.NewMark.delta/(Calc.Solver.NewMark.beta*Calc.Solver.dt)*Veh(veh_num).SysM.C;
        
        % -- Force Matrix --
        F_ext = (Veh(veh_num).Prop.kTi'*ones(1,num_t_app+1)).*Veh(veh_num).Pos.wheels_h(:,1:num_t_app+1) + ...
                (Veh(veh_num).Prop.cTi'*ones(1,num_t_app+1)).*Veh(veh_num).Pos.wheels_hd(:,1:num_t_app+1);
        F_ext = Veh(veh_num).SysM.N2w'*F_ext;

        % -- Step by step calculation --
        for t = 1:num_t_app-1

            % ---- Vehicle System ----
            % Newmark-beta scheme (As seen in B014)
            A = Sol.Veh(veh_num).U(:,t)/(Calc.Solver.NewMark.beta*Calc.Solver.dt^2) + ...
                    Sol.Veh(veh_num).V(:,t)/(Calc.Solver.NewMark.beta*Calc.Solver.dt) + ...
                    Sol.Veh(veh_num).A(:,t)*(1/(2*Calc.Solver.NewMark.beta)-1);
            B = (Calc.Solver.NewMark.delta/(Calc.Solver.NewMark.beta*Calc.Solver.dt)*Sol.Veh(veh_num).U(:,t) - ...
                    (1-Calc.Solver.NewMark.delta/Calc.Solver.NewMark.beta)*Sol.Veh(veh_num).V(:,t) - ...
                    (1-Calc.Solver.NewMark.delta/(2*Calc.Solver.NewMark.beta))*Calc.Solver.dt*Sol.Veh(veh_num).A(:,t));
            Sol.Veh(veh_num).U(:,t+1) = eff_K_veh\(F_ext(:,t+1) + Veh(veh_num).SysM.M*A + Veh(veh_num).SysM.C*B);
            Sol.Veh(veh_num).V(:,t+1) = Calc.Solver.NewMark.delta/(Calc.Solver.NewMark.beta*Calc.Solver.dt)*Sol.Veh(veh_num).U(:,t+1) - B;
            Sol.Veh(veh_num).A(:,t+1) = Sol.Veh(veh_num).U(:,t+1)/(Calc.Solver.NewMark.beta*Calc.Solver.dt^2) - A;

        end % for t = 1:num_t_app-1

        % -- Additional Output generation --
        inds = 1:num_t_app+1;
        % Wheel displacements
        Sol.Veh(veh_num).Wheels.U(:,inds) = Veh(veh_num).SysM.N2w*Sol.Veh(veh_num).U(:,inds);
        % Relative wheel displacements
        Sol.Veh(veh_num).Wheels.Urel(:,inds) = ...
            Sol.Veh(veh_num).Wheels.U(:,inds) - Veh(veh_num).Pos.wheels_h(:,inds);
        % Wheel velocities
        Sol.Veh(veh_num).Wheels.V(:,inds) = Veh(veh_num).SysM.N2w*Sol.Veh(veh_num).V(:,inds);
        % Relative wheel velocities
        Sol.Veh(veh_num).Wheels.Vrel(:,inds) = ...
            Sol.Veh(veh_num).Wheels.V(:,inds) - Veh(veh_num).Pos.wheels_hd(:,inds);
          
    end % if Calc.nt_app > 0

end % for veh_num = 1:Veh(1).Event.num_veh

% ---- End of script ----