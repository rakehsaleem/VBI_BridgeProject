function [Veh] = B25_WheelProfiles(Calc,Veh)

% Calculates the profile under each wheel

% Copyright (c) 2019, Daniel Cantero. All rights reserved.
% Use, copy or distribution of this program is only allowed with the 
% explicit authorization from the author. Please contact daniel.cantero@ntnu.no

% -------------------------------------------------------------------------
%
% -------------------------------- Sketch ---------------------------------
% 
% Max Wheelbase      Approach             Beam          Wheelbase
% |------------|------------------|-----------------|------------|
%                               Calc.x_path(1,:)
%              |-------------------------------------------------|
%                           Calc.x_path(2,:)                 ax_sp(1)
%         |-------------------------------------------------|----|
%                               ...
%                Calc.x
%     |----------------------------------------------------------|
%    Wheelbase
%     |--------|
% -------------------------------------------------------------------------
%
% ---- Input ----
% Calc = Structure with Calculation variable. It should include at least:
% Veh = Structre with Vehicle variables. It should include at least:
% ---- Output ----
% Calc = Additional fields in the structure:
%   .x_path
%   .h_path
%   .hd_path
% -------------------------------------------------------------------------

% Profile for each wheel and vehicle
for veh_num = 1:Veh(1).Event.num_veh

    % 1st derivative for the vehicle's speed
    Calc.Profile.hd = [0,diff(Calc.Profile.h)]/(Calc.Profile.dx/Veh(veh_num).Pos.vel);

    % Initialize
    Veh(veh_num).Pos.wheels_h = Veh(veh_num).Pos.wheels_x*0;
    Veh(veh_num).Pos.wheels_hd = Veh(veh_num).Pos.wheels_h;
    
    % Interpotlation from full length profile 
    %interp_method = 'linear';   % Matlab's default (leads to high frequency artifacts)
    interp_method = 'pchip';    % Piecewise cubic interpolation
    for wheel_num = 1:Veh(veh_num).Prop.num_wheels
        Veh(veh_num).Pos.wheels_h(wheel_num,:) = ...
            interp1(Calc.Profile.x,Calc.Profile.h,Veh(veh_num).Pos.wheels_x(wheel_num,:),interp_method);
        Veh(veh_num).Pos.wheels_hd(wheel_num,:) = ...
            interp1(Calc.Profile.x,Calc.Profile.hd,Veh(veh_num).Pos.wheels_x(wheel_num,:),interp_method);
    end % for wheel_num = 1:Veh(veh_num).Prop.num_wheels
    
end % for veh_num = 1:Veh(1).Event.num_veh

% ---- End of script ----