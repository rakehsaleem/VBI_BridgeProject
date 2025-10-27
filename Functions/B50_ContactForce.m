function [Calc,Sol] = B50_ContactForce(Calc,Veh,Beam,Sol)

% Calculates the vehicle contact forces when crossing the beam

% Copyright (c) 2019, Daniel Cantero. All rights reserved.
% Use, copy or distribution of this program is only allowed with the 
% explicit authorization from the author. Please contact daniel.cantero@ntnu.no

% -------------------------------------------------------------------------

% Deformation under each wheel
[Sol] = B17_Calc_U_at(Calc,Veh,Beam,Sol);

% Force on Beam (Contact Force)
for veh_num = 1:Veh(1).Event.num_veh

    % On bridge time indices
    inds = Calc.Solver.t0_ind_beam:Calc.Solver.t_end_ind_beam;

    % Addition under wheel deformation to original profile --
    Sol.Veh(veh_num).Under.h = Veh(veh_num).Pos.wheels_h(:,inds) + Sol.Veh(veh_num).Under.def*(Calc.Opt.VBI);
    Sol.Veh(veh_num).Under.hd = Veh(veh_num).Pos.wheels_hd(:,inds) + Sol.Veh(veh_num).Under.vel*(Calc.Opt.VBI);

    % Wheels displacements
    Sol.Veh(veh_num).Wheels.U(:,inds) = Veh(veh_num).SysM.N2w*Sol.Veh(veh_num).U(:,inds);
    % Relative wheel displacements
    Sol.Veh(veh_num).Wheels.Urel(:,inds) = ...
        Sol.Veh(veh_num).Wheels.U(:,inds) - Sol.Veh(veh_num).Under.h;
    % Wheel velocities
    Sol.Veh(veh_num).Wheels.V(:,inds) = Veh(veh_num).SysM.N2w*Sol.Veh(veh_num).V(:,inds);
    % Relative wheel velocities
    Sol.Veh(veh_num).Wheels.Vrel(:,inds) = ...
        Sol.Veh(veh_num).Wheels.V(:,inds) - Sol.Veh(veh_num).Under.hd;

    % Force on Beam: Initialized with the static force
    Sol.Veh(veh_num).Under.onBeamF = Veh(veh_num).Static.load*ones(1,Calc.Solver.num_t_beam);

    % Force on Beam: Addition of tyre properties contributions
    Sol.Veh(veh_num).Under.onBeamF = Sol.Veh(veh_num).Under.onBeamF + ...
        (Veh(veh_num).Prop.kTi'*ones(1,Calc.Solver.num_t_beam)).*Sol.Veh(veh_num).Wheels.Urel(:,inds) + ...
        (Veh(veh_num).Prop.cTi'*ones(1,Calc.Solver.num_t_beam)).*Sol.Veh(veh_num).Wheels.Vrel(:,inds);

end % for veh_num = 1:Veh(1).Event.num_veh

% % -- Graphical check of different contributions to the contact force --
% veh_num = 1;
% [Sol] = B17_Calc_U_at(Calc,Veh,Beam,Sol);
% xdata_t = Calc.Solver.t;
% xdata_t_beam = Calc.Solver.t_beam;
% ones_1_x_num_t = ones(1,Calc.Solver.num_t);
% ones_1_x_num_t_beam = ones(1,Calc.Solver.num_t_beam);
% figure;
%     subplot(3,2,1); 
%         plot(xdata_t,((Veh(veh_num).SysM.N2w * Sol.Veh(veh_num).U).* (Veh(veh_num).Prop.kTi' * ones_1_x_num_t))');
%         axis tight; xlim(xdata_t([1,end]));
%         title('due to Veh. def');
%         ylabel('Force (N)');
%     subplot(3,2,3);
%         plot(xdata_t_beam,(-Sol.Veh(veh_num).Under.def.* (Veh(veh_num).Prop.kTi' * ones_1_x_num_t_beam))');
%         axis tight; xlim(xdata_t([1,end]));
%         title('due to Beam def');
%         ylabel('Force (N)');
%     subplot(3,2,5); 
%         plot(((- Veh(veh_num).Pos.wheels_h).* (Veh(veh_num).Prop.kTi' * ones_1_x_num_t))');
%         axis tight; xlim(xdata_t([1,end]));
%         title('due to Profile');
%         ylabel('Force (N)');
%     subplot(3,2,2); 
%         plot(xdata_t,((Veh(veh_num).SysM.N2w * Sol.Veh(veh_num).V).* (Veh(veh_num).Prop.cTi' * ones_1_x_num_t))');
%         axis tight; xlim(xdata_t([1,end]));
%         title('due to Veh. 1st deriv.');
%         ylabel('Force (N)');
%     subplot(3,2,4); 
%         plot(xdata_t_beam,(- Sol.Veh(veh_num).Under.vel .* (Veh(veh_num).Prop.cTi' * ones_1_x_num_t_beam))');
%         axis tight; xlim(xdata_t([1,end]));
%         title('due to Beam 1st deriv.');
%         ylabel('Force (N)');
%     subplot(3,2,6); 
%         plot(((- Veh(veh_num).Pos.wheels_hd) .* (Veh(veh_num).Prop.cTi' * ones_1_x_num_t))');
%         axis tight; xlim(xdata_t([1,end]));
%         title('due to Profile 1st derv.');
%         ylabel('Force (N)');

% ---- End of script ----