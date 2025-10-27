function [Veh] = DOF_2(Veh)
% Vehicle of 2-DOF
% Created manually: 02-Apr-2019
 
% Copyright (c) 2019, Daniel Cantero. All rights reserved.
% Use, copy or distribution of this program is only allowed with the
% explicit authorization from the author. Please contact daniel.cantero@ntnu.no

% -- Vehicle information --
% num_axles_per_body = [1];
% num_axles_per_group = [1];
% with_articulation = [0];
 
% -- Vehicle variables --
% Veh.Prop.mBi = [mB1]
% Veh.Prop.kTi = kT1
% Veh.Prop.cTi = cT1
% Veh.Prop.kSi = kS1
% Veh.Prop.cSi = cS1
% Veh.Prop.mSi = mS1
 
% -- Degrees of Freedom --
Veh.DOF(1).name = 'yB1'; Veh.DOF(1).type = 'displacement'; Veh.DOF(1).dependency = 'independent';
Veh.DOF(2).name = 'yS1'; Veh.DOF(2).type = 'displacement'; Veh.DOF(2).dependency = 'independent';
Veh.DOF(1).num_independent = 2;
 
% -- DOF relations -- 
Veh.DOF(1).num_dependent = 0;

% -- Axle spacing and distance --
Veh.Prop.ax_sp = 0;
Veh.Prop.ax_dist = 0;
 
% -- Vehicle system matrices --
Veh.SysM.M = ...
    [[Veh.Prop.mBi(1), 0]; ...
    [0, Veh.Prop.mSi(1)]];
 
Veh.SysM.C = ...
    [[Veh.Prop.cSi(1), -Veh.Prop.cSi(1)]; ...
    [-Veh.Prop.cSi(1), Veh.Prop.cSi(1) + Veh.Prop.cTi(1)]];
 
Veh.SysM.K = ...
    [[Veh.Prop.kSi(1), -Veh.Prop.kSi(1)]; ...
    [-Veh.Prop.kSi(1), Veh.Prop.kSi(1) + Veh.Prop.kTi(1)]];
 
% -- Force vector to calculate static response --
% Note: When using this vector, multiply it by the gravity. Following the sign criteria defined here
%   gravity has negative value. The numerical value to use is grav = -9.81 m/s^2
Veh.Static.F_vector_no_grav = [Veh.Prop.mBi(1), Veh.Prop.mSi(1)];
 
% -- Nodal disp. to wheel disp. relation --
Veh.SysM.N2w = [0, 1];

% ---- End of function ----