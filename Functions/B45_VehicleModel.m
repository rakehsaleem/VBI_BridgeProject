function [Veh] = B45_VehicleModel(Calc,Veh)

% Runs corresponding vehicle model script to obtain vehicle's system matrices
% Also calculates the static load of each vehicle

% Copyright (c) 2019, Daniel Cantero. All rights reserved.
% Use, copy or distribution of this program is only allowed with the 
% explicit authorization from the author. Please contact daniel.cantero@ntnu.no

% -------------------------------------------------------------------------
% % ---- Inputs ----
% Veh(veh_num) = Structure with Vehicle's variables, including at least:
%   .Model.type = Name of function to run to generate the vehicle's system matrices
%   .Model.function_path = Pathname of the location of the function
%   .Event.num_veh = number of vehicles i the event
% % ---- Outputs ----
% Veh(veh_num) = Addition of fields to structure Veh:
%   .SysM = Mass matrix
%   .SysC = Damping matrix
%   .SysK = Stiffness matrix
%   ... and more
%   .Static.load = Static load of vehicle on the road
% -------------------------------------------------------------------------

% ---- Vehicle system matrices ----

% Current directory
old_dir = cd;

% Move workspace to new directory
cd(Veh(1).Model.function_path)
copy_function_path = Veh(1).Model.function_path;

% Vehicle loop
for veh_num = 1:Veh(1).Event.num_veh
    
    % Running appropriate function
    if veh_num == 1 
        CopyVeh = feval(Veh(veh_num).Model.type,Veh(veh_num));
    else
        CopyVeh(veh_num) = feval(Veh(veh_num).Model.type,Veh(veh_num));
    end % if veh_num == 1 

end % for veh_num = 1:Veh(1).Event.num_veh

% Changing back to original path
cd(old_dir);

% Generating output
Veh = CopyVeh;
Veh(1).Model.function_path = copy_function_path;

% ---- Vehicle static load ----

% Vehicle loop
for veh_num = 1:Veh(1).Event.num_veh
    
    % Calculation of static deformation
    disp = (Veh(veh_num).Static.F_vector_no_grav*Calc.Cte.grav)/Veh(veh_num).SysM.K;

    % Contact Forces
    Veh(veh_num).Static.load = Veh(veh_num).Prop.kTi'.*(Veh(veh_num).SysM.N2w*disp');
    
    % Checks
    Veh(veh_num).Static.check = ...
        (sum(Veh(veh_num).Static.load)-((sum(Veh(veh_num).Prop.mBi)+sum(Veh(veh_num).Prop.mSi))*Calc.Cte.grav))<Calc.Cte.tol;
    if Veh(veh_num).Static.check==false
        disp(['Error: Vehicle ',num2str(veh_num),' -> in calculation of static contact force']);
    end % if Veh(veh_num).Static.check==false

end % for veh_num = 1:Veh(1).Event.num_veh

% ---- End of function ----