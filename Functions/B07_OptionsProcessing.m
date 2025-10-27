function [Calc,Veh,Beam] = B07_OptionsProcessing(Calc,Veh,Beam)

% Processing of input variables, and generating new or changing auxiliary 
% variables accordingly. Definition of default values. Performing some checks 
% on the inputs

% Copyright (c) 2019, Daniel Cantero. All rights reserved.
% Use, copy or distribution of this program is only allowed with the 
% explicit authorization from the author. Please contact daniel.cantero@ntnu.no

% -------------------------------------------------------------------------
% ---- Input ----
% See A00
% ---- Output ----
% See A00
% -------------------------------------------------------------------------

% ---- Veh structure variable ----

% Number of vehicle in the event
Veh(1).Event.num_veh = size(Veh,2);

% Signs of velocity and x0 should be opposite
for veh_num = 1:Veh(1).Event.num_veh
    if sign(Veh(veh_num).Pos.x0)==sign(Veh(veh_num).Pos.vel)
        error(['Error: Vehicle ',num2str(veh_num),' -> the sign of either ''x0'' or ''vel'' is wrong']);
    end % if sign(Veh(veh_num).Pos.x0)==sign(Veh(veh_num).Pos.vel)
end % for veh_num = 1:Veh(1).Event.num_veh

% Maximum velocity in event
Veh(1).Event.max_vel = abs(Veh(1).Pos.vel);
for veh_num = 2:Veh(1).Event.num_veh
    Veh(1).Event.max_vel = max(Veh(1).Event.max_vel,abs(Veh(veh_num).Pos.vel));
end % for veh_num = 2:Veh(1).Event.num_veh

% Default path name for vehicle functions
if ~myIsfield(Veh(1),{'Model',1,'function_path'})
    Veh(1).Model.function_path = 'Vehicle_equations';
end % if ~myIsfield(Veh(1),{'Model',1,'function_path'})

% ---- Calc structure variable ----

% Default value of gravity
if ~myIsfield(Calc,{'Cte',1,'grav'})
    Calc.Cte.grav = -9.81;          % Gravity [m/s^2]
end % if ~myIsfield(Calc,{'Cte',1,'grav'})

% Beam Damage Type
if ~isfield(Beam,'Damage')
    Beam.Damage.type = 0;       % No beam damage considered
end % if ~isfield(Beam,'Damage')

% Default calculation of Beam natural frequencies
if ~myIsfield(Calc,{'Opt',1,'beam_frq'})
    Calc.Opt.beam_frq = 1;
end % if ~myIsfield(Calc,{'Opt',1,'beam_frq'})

% Default calculation of Beam modes of vibration
if ~myIsfield(Calc,{'Opt',1,'beam_modes'})
    Calc.Opt.beam_frq = 1;
    Calc.Opt.beam_modes = 1;
end % if ~myIsfield(Calc,{'Opt',1,'beam_modes'})

% If no plots are selected
if ~isfield(Calc,'Plot')
    Calc.Plot.NoPlot = 1;
end % if ~isfield(Calc,'Plot')

% Default calculation of Vehicle natural frequencies
if ~isfield(Calc.Opt,'veh_frq')
    Calc.Opt.veh_frq = 1;
end % if ~isfield(Calc.Opt,'veh_frq')

% Default value of numerical tolerance
if ~isfield(Calc.Cte,'tol')
    Calc.Cte.tol = 1e-6;
end % if ~isfield(Calc.Cte,'tol')

% Definition of accurate mode inclusion
if ~myIsfield(Calc,{'Solver',1,'min_Beam_modes_considered'})
    Calc.Solver.min_Beam_modes_considered = 1;
end % if ~myIsfield(Calc,{'Solver',1,'min_Beam_modes_considered'})

% Default value of solver's maximum accurate frequency
if ~myIsfield(Calc,{'Solver',1,'max_accurate_frq'})
    Calc.Solver.max_accurate_frq = 0;
end % if ~myIsfield(Calc,{'Solver',1,max_accurate_frq})

% Definition of minimum time steps per second
if ~isfield(Calc.Solver,'min_t_steps_per_second')
    Calc.Solver.min_t_steps_per_second = 0;
end % if ~isfield(Calc.Solver,'min_t_steps_per_second')

% Loading profile option
if Calc.Profile.type == -1
    Calc.Profile.Load.on = 1;
else
    Calc.Profile.Load.on = 0;
end % if Calc.Profile.type == -1

% Maximum profile spatial frequencies 
if Calc.Profile.type == 2
    % Also minimum and reference Freq. in case of ISO profile
    Calc.Profile.Spatial_frq.min = 0.01;
    Calc.Profile.Spatial_frq.max = 4;          % Arturo's criteria
    Calc.Profile.Spatial_frq.ref = 0.1;
else
    Calc.Profile.Spatial_frq.max = 0;
end % if Calc.Profile.type == 2

% Checking definition of Profile filename to load
if Calc.Profile.Load.on == 1
    if ~isfield(Calc.Profile.Load,'file_name')
        error('Error: Profile definition -> Profile is to be loaded but no file name is defined')
    end % ~isfield(Calc.Profile.Load,'file_name')
end % if Calc.Profile.Load.on == 1

% Default loading path
if Calc.Profile.Load.on == 1
    if ~isfield(Calc.Profile.Load,'path')
        Calc.Profile.Load.path = 'Profiles\';
    end % if ~isfield(Calc.Profile.Load,'path')
end % if Calc.Profile.Load.on == 1

% Default value of Save.on
if ~myIsfield(Calc.Profile,{'Save',1,'on'})
    Calc.Profile.Save.on = 0;
end % if ~myIsfield(Calc.Profile,{'Save',1,'on'})

% Default saving path
if Calc.Profile.Save.on == 1
    if ~isfield(Calc.Profile.Save,'path')
        Calc.Profile.Save.path = 'Profiles\';
    end % if ~isfield(Calc.Profile.Save,'path')
end % if Calc.Profile.Save.on == 1

% Checking definition of Profile filename to save
if Calc.Profile.Save.on == 1
    if ~isfield(Calc.Profile.Save,'file_name')
        error('Error: Profile definition -> Profile is to be saved but no file name is defined')
    end % ~isfield(Calc.Profile.Save,'file_name')
end % if Calc.Profile.Save.on == 1

% Checking incompatibility of "Save.on" and "Load.on"
if Calc.Profile.Load.on == 1
    if Calc.Profile.Save.on == 1
        error('Error: Profile load/save -> Profile should not be loaded and saved in the same simulation');
    end % if Calc.Profile.Save.on == 1
end % if Calc.Profile.Load.on == 1

% Default profile sampling distance (m)
if ~isfield(Calc.Profile,'dx')
    Calc.Profile.dx = 0.01;
end % if ~isfield(Calc.Profile,'dx')

if Calc.Profile.type == 2
    % ISO profiles classification limits
    % Limits between [minA,A-B,B-C,C-D,D-E,E-F,F-G,G-H]
    %Calc.Profile.Gd_limits = [8,32,128,512,2048,8192,32768,131072]
    Calc.Profile.Info.Gd_limits = (2.^(3:2:17))*1e-6;
    % A = Very Good; (8e-6 <=) Gd < 32e-6
    % B = Good; 32e-6 <= Gd < 128e-6
    % C = Average; 128e-6 <= Gd < 512e-6
    % D = Poor; 512e-6 <= Gd < 2048e-6
    % E = Very Poor; 2048e-6 <= Gd < 8192e-6
    % F = 8192e-6 <= Gd < 32768e-6
    % G = 32768e-6 <= Gd < 131072e-6
    % H = 131072e-6 <= Gd

    % Default value for Calc.Profile.class_var = 0;
    if ~myIsfield(Calc.Profile,{'Opt',1,'class_var'})
        Calc.Profile.Opt.class_var = 0;
    end % if ~myIsfield(Calc.Profile,{'Opt',1,'class_var'})
    
end % if Calc.Profile.type == 2

% Default window length for moving average filter
if myIsfield(Calc.Profile,{'Opt',1,'movAvg'},1)
    if ~myIsfield(Calc.Profile,{'Opt',1,'window_L'})
        Calc.Profile.Opt.window_L = 0.24; % [m]
    end % if ~myIsfield(Calc.Profile,{'Opt',1,'window_L'})
end % if myIsfield(Calc.Profile,{'Opt',1,'movAvg'},1)

% Default profile length
if ~isfield(Calc.Profile,'L')
    Calc.Profile.L = 1000; % [m]
end % if ~isfield(Calc.Profile,'L')

% Verbose On/Off
if ~isfield(Calc.Opt,'verbose')
    Calc.Opt.verbose = 0;
end % if isfield(Calc,'verbose')

% Verbose On/Off
if ~isfield(Calc.Opt,'verbose')
    Calc.Opt.verbose = 0;
end % if isfield(Calc,'verbose')

% Show progress
if ~isfield(Calc.Opt,'show_progress_every_s')
    Calc.Opt.show_progress_every_s = 2;
end % if ~isfield(Calc.Opt,'show_progress_every_s')

% Default value for calculation of vehicles initial static deformations
if ~isfield(Calc.Opt,'vehInitSta')
    Calc.Opt.vehInitSta = 1;
end % if ~isfield(Calc.Opt,'vehInitSta')

% Newmark-Beta scheme
% Default: Average acceleration method (Normal Newmark-beta)
if ~myIsfield(Calc.Solver,{'NewMark',1,'damp'})
    Calc.Solver.NewMark.damp = 0;      
end % if ~myIsfield(Calc.Solver,{'NewMark',1,'damp'})

% Newmark scheme (for vehicle and beam solvers)
if Calc.Solver.NewMark.damp == 0
    % Default = Average (or constant) acceleration method
    Calc.Solver.NewMark.delta = 0.5; 
    Calc.Solver.NewMark.beta = 0.25;
elseif Calc.Solver.NewMark.damp == 1
    % Damped Newmark-Beta scheme
    Calc.Solver.NewMark.delta = 0.6; 
    Calc.Solver.NewMark.beta = 0.3025;
end % if Calc.Solver.NewmMark.damp

% Procedure for interaction solution
% Selection of procedure
if ~myIsfield(Calc,{'Proc',1,'name'})
    error('No solving procedure selected!');
end % if ~myIsfield(Calc,{'Proc',1,'name'})
% Procedure code
if strcmp(Calc.Proc.name,'FI')
    Calc.Proc.code = 1;
elseif strcmp(Calc.Proc.name,'SSI')
    Calc.Proc.code = 2;
elseif strcmp(Calc.Proc.name,'Coup')
    Calc.Proc.code = 3;
end % if strcmp(Calc.Proc.name,'FI');

% Initialize iteration counter
if Calc.Proc.code == 1
    if ~isfield(Calc.Proc.Iter,'num')
        Calc.Proc.Iter.num = 0;
    end % if ~isfield(Calc.Proc.Iter,'num')
% elseif Calc.Proc.code == 2    % Done in B48
%     if ~isfield(Calc.Proc.Iter,'num_t_bri')
%         Calc.Proc.Iter.num_t_bri = 0;
%     end % if ~isfield(Calc.Proc.Iter,'num')
end % if Calc.Proc.code == 1

% FI default Iteration variables
if Calc.Proc.code == 1
    % Reset logical flag
    if ~myIsfield(Calc,{'Proc',1,'Iter',1,'continue'})
        Calc.Proc.Iter.continue = 1;
    end % if ~myIsfield(Calc,{'Proc',1,'Iter',1,'continue'})
end % if Calc.Proc.code == 1

% FI and SSI default Iteration variables
if or(Calc.Proc.code == 1,Calc.Proc.code == 2)

    % Default maximum number of iterations
    if ~isfield(Calc.Proc.Iter,'max_num')
        Calc.Proc.Iter.max_num = 10;
    end % if ~isfield(Calc.Proc.Iter,'max_num');

    % Default Iterative criteria
    if ~isfield(Calc.Proc.Iter,'criteria')
        Calc.Proc.Iter.criteria = 1;         % Iteration criteria based on deformation under wheels
        %Calc.Proc.Iter.criteria = 2;         % Iteration criteria based on BM of whole beam
    end % if ~isfield(Calc.Proc.Iter,'criteria')

    % Iteration criteria text label
    if Calc.Proc.Iter.criteria == 1
        Calc.Proc.Iter.criteria_text = 'Beam deformation under wheels';
    elseif Calc.Proc.Iter.criteria == 2
        Calc.Proc.Iter.criteria_text = 'Whole beam BM';
    end % if Calc.Proc.Iter.criteria == 1

    % Default Iterative process tolerance for stopping criteria
    if ~isfield(Calc.Proc.Iter,'tol')
        Calc.Proc.Iter.tol = Calc.Cte.tol;
    end % if ~isfield(Calc.Proc.Iter,'tol')

% Coupled default values
elseif Calc.Proc.code == 3
    
    
end % if or(Calc.Proc.code == 1,Calc.Proc.code == 2)

% Default value for calculation of initial beam static deformation (Default = 0)
if ~isfield(Calc.Opt,'beamInitSta')
    Calc.Opt.beamInitSta = 1;      % Calculate Beam initial static deformation
end % if ~isfield(Calc.Opt,'beamInitSta')

% Default BM calculation method
if ~isfield(Calc.Opt,'calc_mode_BM')
    Calc.Opt.calc_mode_BM = 1;      % The average nodal stress is considered
    %Calc.Opt.calc_mode_BM = 0;      % No average value
end % if ~isfield(Calc.Opt,'calc_mode_BM')

% Default Shear calculation method
if ~isfield(Calc.Opt,'calc_mode_Shear')
    Calc.Opt.calc_mode_Shear = 1;      % The average nodal stress is considered
    %Calc.Opt.calc_mode_Shear = 0;      % No average value
end % if ~isfield(Calc.Opt,'calc_mode_Shear')

% Switch ON/OFF VBI
if ~isfield(Calc.Opt,'VBI')
    Calc.Opt.VBI = 1;   % VBI is ON
elseif Calc.Opt.VBI == 0
    Calc.Proc.Iter.max_num = 0;
end % if ~isfield(Calc.Opt,'VBI')

% Additional free vibration seconds to calculate beam response
if ~isfield(Calc.Opt,'free_vib_seconds')
    Calc.Opt.free_vib_seconds = 0;
end % if ~isfield(Calc.Opt,'free_vib_seconds')

% Include Coriolis effect
if ~isfield(Calc.Opt,'include_coriolis')
    Calc.Opt.include_coriolis = 1;
end % if ~isfield(Calc.Opt,'include_coriolis')

% ---- Beam structure variable ----

% Beam Damping
if isfield(Beam.Prop,'damp_per')
    Beam.Prop.damp_xi = Beam.Prop.damp_per/100;
else
    Beam.Prop.damp_per = 0;
    Beam.Prop.damp_xi = 0;
end % if isfield(Beam,'damp_per')

% Definition of mid-span index
if mod(Beam.Mesh.Ele.num+1,2) == 0
    disp('No node defined at the mid-span of the Beam')
    disp('Please choose an even number elements')
    error('Please choose an even number elements');
else
    Beam.Mesh.Node.at_mid_span = (Beam.Mesh.Ele.num)/2+1;
end % if mod(Beam.Mesh.Ele.num,2) == 0

% Mass Matric consistency
% Consistent Mass Matrix (CMM) = 1; (Default)
% Lumped Mass Matrix (LMM) = 0;
% Other values between [0,1] are also possible
if ~myIsfield(Beam,{'SysM',1,'Opt',1,'Mconsist_value'})
    Beam.SysM.Opt.Mconsist_value = 1;
end % if ~myIsfield(Beam,{'SysM',1,'Opt',1,'Mconsist_value'})

% ---- End of function ----