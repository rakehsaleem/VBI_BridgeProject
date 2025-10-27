% ---- Moving Vehicle over FEM Beam ----

% General script to run VBI_DC_v2019

% Copyright (c) 2019, Daniel Cantero. All rights reserved.
% Use, copy or distribution of this program is only allowed with the 
% explicit authorization from the author. Please contact daniel.cantero@ntnu.no

% % % % -------------------------------------------------------------------------%   
%  clear all;
%    close all;
%    clear;
%  clc; rng('shuffle'); addpath 'Functions'; tic


% -------------------------------------------------------------------------
% ------------------------- Model Properties ------------------------------
% Span length
% Beam.Prop.Lb = 16+29+29+15;
% % Modulus of elasticity (N/m^2)
% Beam.Prop.E = 3.5e10;
% % Density
% Beam.Prop.rho = 2500;
% 
% % Section properties
% width = 12.19;  % Actual bridge widht 8.6
% thickness = 1.91;
% flange_thickness = 0.16;
% web_width = 0.18*6;
% 
% I_factor = 0.40;
% A_factor = 1;
% 
% %Following this formula:
% %https://amesweb.info/section/second-moment-of-area-calculator.aspx
% h = flange_thickness;
% b = web_width;
% B = width;
% H = thickness-2*h;
% % ------------------------- Beam properties -------------------------------
% Beam.Prop.I = H^3*b/12+2*(h^3*B/12+h*B*(H+h)^2/4)
% Beam.Prop.I = Beam.Prop.I*I_factor;
% Beam.Prop.A = width*thickness-(thickness-2*flange_thickness)*(width-web_width);
% Beam.Prop.A = Beam.Prop.A*A_factor
% % ------------------------- Beam properties -------------------------------
% 
% % -- Properties --
% Beam.Prop.type = 0;             % Custom definition of beam properties    %(Bridge I lb=28125, 1.2985(need to be find) 0.8
% Beam.Prop.Lb =  15;             % Beam's length
% Beam.Prop.rho = 28125 %18358 % 28125  %;   %       % Density per meter length %  1.3901;
% Beam.Prop.E = 3.5e10;           % E
% Beam.Prop.I = 0.5273  % %1.3901 %1.2985      %0.5273    % I
% Beam.Prop.A =  0.75 %0.8 %0.75;      %0.8          % Area
Beam.Prop.damp_per =0;         % Damping percentage
Beam.Prop.h = 1;                % Beam height 
% % 
Beam.Prop.type = 'T';             % Custom definition of beam properties    %(Bridge I lb=28125, 1.2985(need to be find) 0.8
Beam.Prop.Lb =  15; 
% 
% % -- Boundary conditions --
% % Beam.BC.loc = [0,Beam.Prop.Lb];       % Location of supports
% %Beam.BC.vert_stiff = [-1,-1];          % Vertical stiffness (0 = free, -1 = fixed, other = vertical stiffness)
% % Beam.BC.rot_stiff = [-1,-1];          % Rotational stiffness (0 = free, -1 = fixed, other = rotational stiffness)
% % Pinned-Pinned
%   Beam.BC.loc = [0,Beam.Prop.Lb]; Beam.BC.vert_stiff = [-1,-1]; Beam.BC.rot_stiff = [0,0];
% % Fixed-Fixed
  Beam.BC.loc = [0,Beam.Prop.Lb]; Beam.BC.vert_stiff = [-1,-1]; Beam.BC.rot_stiff = [-1,-1];
% % % With some rotational stiffness
%  %Beam.BC.loc = [0,Beam.Prop.Lb]; Beam.BC.vert_stiff = [-1,-1]; Beam.BC.rot_stiff = [1,1]*1e10;
% % % Two span bridge (pin supports)
% %Beam.BC.loc = [0,Beam.Prop.Lb/2,Beam.Prop.Lb]; Beam.BC.vert_stiff = [-1,-1,-1]; Beam.BC.rot_stiff = [0,0,0];
% Beam.BC.vert_stiff = [-1,2.5e10,2.5e10*0.7,2.5e10,-1]; Beam.BC.rot_stiff = [0,1,0.7,1,0]*4.5e9;
% Beam.BC.loc = [0,16,16+29,16+29*2,16+29*2+15]; 
% -- Mesh --
Beam.Mesh.Ele.num = Beam.Prop.Lb*2 % Number of elements (Should be an even number)

% -- Damage --
%Beam.Damage.type =1;           % No damage (default value)
%Beam.Damage.type = 1;           % 1 element damage
Beam.Damage.type =0;          % Global damage
Beam.Damage.E.per = [66,45];    % Stiffnes reduction - [Location, Magnitude] of damage in [%]
%Beam.Damage.E.per = [13,50];
%Beam.Damage.I.per = [28,50];   % Inertia reduction - [Location, Magnitude] of damage in [%]
%Beam.Damage.rho.per = [13,30]; % Density reduction - [Location, Magnitude] of damage in [%]
%Beam.Damage.A.per = [12,50];   % Area reduction - [Location, Magnitude] of damage in [%]
%Beam.Damage.h.per = [50,20];   % Area reduction - [Location, Magnitude] of damage in [%]

% ----------------------------- Profile -----------------------------------

% -- Type --
  Calc.Profile.type = -1;                             % To load an existing proile
        Calc.Profile.Load.file_name ='P02_Profile_Class_A' %GD=2.00E05 'P03_Profile_Class_A'
 %  Calc.Profile.type = 0;               % Smooth profile
%     Calc.Profile.type = 2;               % ISO random profile 
%           Calc.Profile.Info.class = 'A';   % ISO class in text format
%           Calc.Profile.Opt.class_var = 0;  % Consider variation 
   %   Calc.Profile.Opt.class_var = 1; % Consider variation of roughness within the profile class
%Calc.Profile.type = 3;               % Step profile
   %Calc.Profile.step_loc_x = -100;    % Step location in x-coordinate system [m]
   %Calc.Profile.step_height = 0.001/2; % Step height in [m]
 %Calc.Profile.type = 4;               % Ramp profile
  %  Calc.Profile.ramp_loc_x_0 = -100;  % Ramp start location in x-coordinate system [m]
   % Calc.Profile.ramp_loc_x_end = 1;  % Ramp end location in x-coordinate system [m]    
    %Calc.Profile.ramp_height = -0.01; % Ramp height in [m]
  % Calc.Profile.type = 5;               % Sinewave profile
   %    Calc.Profile.sine_wavelength = 2;% Wavelength of the sinusoidal [m]
    %   Calc.Profile.sin_Amp = 0.005;      % Amplitude of the sinusoidal [m]
     %  Calc.Profile.sin_phase = 0;       % Phase of sinusoidal at x=0 [rad]

% -- Saving Profile --
%      Calc.Profile.Save.on = 1;
%      Calc.Profile.Save.file_name = 'P03_Profile_Class_A_Long_bridge_New';

% -- Other Options --
Calc.Profile.Opt.movAvg = 1;   % Moving Average filter for profile 
  Calc.Profile.L = 1000;         % Profile length (longer than necessary)

% ------------------------- Vehicle Properties ----------------------------

veh_num = 0;

% -- 1-DOF vehicle -- (Sprung mass)
% veh_num = veh_num + 1;
% Veh(veh_num).Model.type = 'DOF_1';      % Text with name of model to use
% Veh(veh_num).Prop.mSi = 1*1e3;          % Suspension mass [kg]
% Veh(veh_num).Prop.kTi = 1*1e6;          % Tyre stiffness [N/m]
% Veh(veh_num).Prop.cTi = 1*1e4;          % Tyre viscous damping [N*s/m]
% Veh(veh_num).Pos.vel = 12;              % Speed of vehicle [m/s] (negative value = vehicle moving from right to left)
% Veh(veh_num).Pos.x0 = -100;             % Initial position of vehicle (from left bridge support)

%%-- 2-DOF vehicle -- (Q-car)
% veh_num = veh_num + 1;
% Veh(veh_num).Model.type = 'DOF_2';      % Text with name of model to use
% Veh(veh_num).Prop.mBi = 1400;               % Body masses [kg]
% Veh(veh_num).Prop.kSi = 400*1e3;        % Suspension stiffness [N/m]
% Veh(veh_num).Prop.cSi = 30*1e3;         % Suspension viscous damping [N*s/m]
% Veh(veh_num).Prop.mSi = 0.7*1e3;        % Suspension mass [kg]
% Veh(veh_num).Prop.kTi = 1.75*1e6;       % Tyre stiffness [N/m]
% Veh(veh_num).Prop.cTi = 0*1e4;          % Tyre viscous damping [N*s/m]
% Veh(veh_num).Pos.vel =  12; %Veh(1).Pos.vel;  % Speed of vehicle [m/s] (negative value = vehicle moving from right to left)
% Veh(veh_num).Pos.x0 =   -100;           % Initial position of vehicle (from left bridge support)
% Veh(veh_num).Pos.vel = -20;
% Veh(veh_num).Pos.x0 = 125;
% 
%-- 2-DOF vehicle -- (Q-car)
% veh_num = veh_num + 1;
% Veh(veh_num).Model.type = 'DOF_2';      % Text with name of model to use
% Veh(veh_num).Prop.mBi = Veh(1).Prop.mb;           % Body masses [kg]
% Veh(veh_num).Prop.kSi = 4*1e5;          % Suspension stiffness [N/m]
% Veh(veh_num).Prop.cSi = 1e4;            % Suspension viscous damping [N*s/m]
% Veh(veh_num).Prop.mSi = Veh(1).Prop.msi;      % Suspension mass [kg]
% Veh(veh_num).Prop.kTi = Veh(1).Prop.Kti       % Tyre stiffness [N/m]
% Veh(veh_num).Prop.cTi = 0;              % Tyre viscous damping [N*s/m]
% Veh(veh_num).Pos.vel = Veh(1).Pos.vel;  % Speed of vehicle [m/s] (negative value = vehicle moving from right to left)
% Veh(veh_num).Pos.x0 = -100 ;            % Initial position of vehicle (from left bridge support)
% Veh(veh_num).Pos.vel = -20;
% Veh(veh_num).Pos.x0 = 125;
% % -- Truck 2 --
%  veh_num = veh_num + 1;
% Veh(veh_num).Model.type = 'Truck_2';     % Text with name of model to use
% Veh(veh_num).Prop.mBi = Veh(1).Prop.mb;           % Body masses [kg]
% Veh(veh_num).Prop.IyBi = Veh(1).Prop.Iy;          % Body moments of inertia [kg*m2]
% Veh(veh_num).Prop.kTi = [1.75,1.75]*1e6; % Tyre stiffness [N/m]
% Veh(veh_num).Prop.cTi = [0,0]*1e4;       % Tyre viscous damping [N*s/m]
% Veh(veh_num).Prop.kSi = [Veh(1).Prop.Ksi,Veh(1).Prop.Ksi];   % Suspension stiffness [N/m]
% Veh(veh_num).Prop.cSi = [1,1]*1e4;       % Suspension viscous damping [N*s/m]
% Veh(veh_num).Prop.mSi = [Veh(1).Prop.msi,Veh(1).Prop.msi];   % Suspension mass [kg]
% Veh(veh_num).Prop.ei = [-3, 0];          % Coordinate of axle (group) to centre of gravity of its corresponding body [m]
% %Veh(veh_num).Pos.vel = -20;             % Speed of vehicle [m/s] (negative value = vehicle moving from right to left)
% %Veh(veh_num).Pos.x0 = 110;              % Initial position of vehicle (from left bridge support)
% Veh(veh_num).Pos.vel = Veh(1).Pos.vel;              % Speed of vehicle [m/s] (negative value = vehicle moving from right to left)
% Veh(veh_num).Pos.x0 = -100;               % Initial position of vehicle (from left bridge support)

% % -- Truck 2 --
% veh_num = veh_num + 1;
% Veh(veh_num).Model.type = 'Truck_2';     % Text with name of model to use
% Veh(veh_num).Prop.mBi = 16000; %Veh(1).Prop.mb;   % Body masses [kg]
% Veh(veh_num).Prop.IyBi = 53651;          % Body moments of inertia [kg*m2]
% Veh(veh_num).Prop.kTi = [1.75,1.75]*1e6;  % Tyre stiffness [N/m]
% Veh(veh_num).Prop.cTi = [0,0]*1e4;       % Tyre viscous damping [N*s/m]
% Veh(veh_num).Prop.kSi = [1,1]*1e6;   % Suspension stiffness [N/m]
% Veh(veh_num).Prop.cSi = [1,1]*1e4;      % Suspension viscous damping [N*s/m]
% Veh(veh_num).Prop.mSi = [1.2,1.2]*1e3;  % Suspension mass [kg]
% Veh(veh_num).Prop.ei = [-8, 0];         % Coordinate of axle (group) to centre of gravity of its corresponding body [m]
% %Veh(veh_num).Pos.vel = -20;            % Speed of vehicle [m/s] (negative value = vehicle moving from right to left)
% %Veh(veh_num).Pos.x0 = 110;             % Initial position of vehicle (from left bridge support)
% Veh(veh_num).Pos.vel = 50 % Veh(1).v;   % Speed of vehicle [m/s] (negative value = vehicle moving from right to left)
% Veh(veh_num).Pos.x0 = -100;             % Initial position of vehicle (from left bridge support)

% veh_num = veh_num + 1;
% Veh(veh_num).Model.type = 'Truck_2';        % Text with name of model to use
% Veh(veh_num).Prop.mBi =  Veh(1).Prop.mbd;   % Body masses [kg]
% Veh(veh_num).Prop.IyBi = Veh(1).Prop.Iy;          % Body moments of inertia [kg*m2]
% Veh(veh_num).Prop.kTi =  [Veh(1).Prop.Kt,Veh(1).Prop.Kt];  % Tyre stiffness [N/m]
% Veh(veh_num).Prop.cTi =  [0,0]*1e4;       % Tyre viscous damping [N*s/m]
% Veh(veh_num).Prop.kSi =  [Veh(1).Prop.Ks,Veh(1).Prop.Ks];      % Suspension stiffness [N/m]
% Veh(veh_num).Prop.cSi =  [Veh(1).Prop.Cs,Veh(1).Prop.Cs];      % Suspension viscous damping [N*s/m]
% Veh(veh_num).Prop.mSi =  [Veh(1).Prop.max,Veh(1).Prop.max];    % Suspension mass [kg]
% Veh(veh_num).Prop.ei =   [-8, 0] %[Veh(1).Prop.Spa, 0];         % Coordinate of axle (group) to centre of gravity of its corresponding body [m]
% %Veh(veh_num).Pos.vel = -20;            % Speed of vehicle [m/s] (negative value = vehicle moving from right to left)
% %Veh(veh_num).Pos.x0 = 110;             % Initial position of vehicle (from left bridge support)
% Veh(veh_num).Pos.vel = 20% Veh(1).Pos.vel;   % Speed of vehicle [m/s] (negative value = vehicle moving from right to left)
% Veh(veh_num).Pos.x0 = -100;             % Initial position of vehicle (from left bridge support)

% % -- Truck 2-2 --
% veh_num = veh_num + 1;
% Veh(veh_num).Model.type = 'Truck_2_2';      % Text with name of model to use
% Veh(veh_num).Prop.mBi = [5000, 10000];      % Body masses [kg]
% Veh(veh_num).Prop.IyBi = [10000, 60000];    % Body moments of inertia [kg*m2]
% Veh(veh_num).Prop.kTi = [1,1,1,1]*1e6;      % Tyre stiffness [N/m]
% Veh(veh_num).Prop.cTi = [0,0,0,0]*1e4;      % Tyre viscous damping [N*s/m]
% Veh(veh_num).Prop.kSi = [1,1,1,1]*1e6;      % Suspension stiffness [N/m]
% Veh(veh_num).Prop.cSi = [1,1,1,1]*1e4;      % Suspension viscous damping [N*s/m]
% Veh(veh_num).Prop.mSi = [1,1,1,1]*1e3;      % Suspension mass [kg]
% Veh(veh_num).Prop.afBi = [0, 2];            % Distance from Body centre of gravity to location of front articulation [m]
% Veh(veh_num).Prop.abBi = 5;                 % Distance from Body centre of gravity to location of back articulation [m]
% Veh(veh_num).Prop.ei = [-1, 3, 1, 2];       % Coordinate of axle (group) to centre of gravity of its corresponding body [m]
% Veh(veh_num).Pos.vel = -20;                 % Speed of vehicle [m/s] (negative value = vehicle moving from right to left)
% Veh(veh_num).Pos.x0 = 100;                  % Initial position of vehicle (from left bridge support)

% %-- Truck 2-3 --
% veh_num = veh_num + 1;
% Veh(veh_num).Model.type = 'Truck_2_3';      % Text with name of model to use
% Veh(veh_num).Prop.mBi = [3200, 10000];      % Body masses [kg]
% Veh(veh_num).Prop.IyBi = [4000,123000];    % Body moments of inertia [kg*m2]
% Veh(veh_num).Prop.kTi = [1.75,1.75,3.5,3.5,3.5]*1e6;      % Tyre stiffness [N/m]
% Veh(veh_num).Prop.cTi = [1,1,2,2,2]*0.5e4;      % Tyre viscous damping [N*s/m]
% Veh(veh_num).Prop.kSi = [6,6,10,10,10]*1e6;      % Suspension stiffness [N/m]
% Veh(veh_num).Prop.cSi = [1,1,2,2,2]*1e4;      % Suspension viscous damping [N*s/m]
% Veh(veh_num).Prop.mSi = [0.550,0.550,0.800,0.800,0.800]*4e3;      % Suspension mass [kg]
% Veh(veh_num).Prop.afBi = [0, 5];            % Distance from Body centre of gravity to location of front articulation [m]
% Veh(veh_num).Prop.abBi = 4;                 % Distance from Body centre of gravity to location of back articulation [m]
% Veh(veh_num).Prop.ei = [-1, 3.5, 1.2, 2.2, 3];  % Coordinate of axle (group) to centre of gravity of its corresponding body [m]
% Veh(veh_num).Pos.vel = 12;                 %  Speed of vehicle [m/s] (negative value = vehicle moving from right to left)
% Veh(veh_num).Pos.x0 = -100;                  % Initial position of vehicle (from left bridge support)
% clear veh_num

% %-- Truck 2-3 -- Truck with fleet
% veh_num = veh_num + 1;
% Veh(veh_num).Model.type = 'Truck_2_3';      % Text with name of model to use
% Veh(veh_num).Prop.mBi =  [Veh(1).Prop.mbf, Veh(1).Prop.mbb];      % Body masses [kg]
% Veh(veh_num).Prop.IyBi = [Veh(1).Prop.Iyf, Veh(1).Prop.Iyb];    % Body moments of inertia [kg*m2]
% Veh(veh_num).Prop.kTi = [1.75,1.75,3.5,3.5,3.5]*1e6;      % Tyre stiffness [N/m]
% Veh(veh_num).Prop.cTi = [1,1,2,2,2]*0.5e4;       % Tyre viscous damping [N*s/m]
% Veh(veh_num).Prop.kSi = [6,6,10,10,10]*1e6;      % Suspension stiffness [N/m]
% Veh(veh_num).Prop.cSi = [1,1,2,2,2]*2e4;      % Suspension viscous damping [N*s/m]
% Veh(veh_num).Prop.mSi = [0.550,0.550,0.800,0.800,0.800]*1e3;      % Suspension mass [kg]
% Veh(veh_num).Prop.afBi = [0, 5];            % Distance from Body centre of gravity to location of front articulation [m]
% Veh(veh_num).Prop.abBi = 4;                 % Distance from Body centre of gravity to location of back articulation [m]
% Veh(veh_num).Prop.ei = [-1, 3.5, 1.2, 2.2, 3];  % Coordinate of axle (group) to centre of gravity of its corresponding body [m]
% Veh(veh_num).Pos.vel = 12; %Veh(1).Pos.vel;                 %  Speed of vehicle [m/s] (negative value = vehicle moving from right to left)
% Veh(veh_num).Pos.x0 = -100;                  % Initial position of vehicle (from left bridge support)
% clear veh_num



% % %-- Truck 2-3 -- Monte-Carlo
veh_num = veh_num + 1;
Veh(veh_num).Model.type = 'Truck_2_3';      % Text with name of model to use
Veh(veh_num).Prop.mBi =  [Veh(1).Prop.mbf, Veh(1).Prop.mbb];      % Body masses [kg]
Veh(veh_num).Prop.IyBi = [Veh(1).Prop.Iyf, Veh(1).Prop.Iyb];    % Body moments of inertia [kg*m2]
Veh(veh_num).Prop.kTi =  [Veh(1).Prop.Ktif, Veh(1).Prop.Ktif, Veh(1).Prop.Ktib, Veh(1).Prop.Ktib, Veh(1).Prop.Ktib];      % Tyre stiffness [N/m]
Veh(veh_num).Prop.cTi =  [1,1,2,2,2]*0.5e4;      % Tyre viscous damping [N*s/m]
Veh(veh_num).Prop.kSi =  [Veh(1).Prop.Ksif,Veh(1).Prop.Ksif,Veh(1).Prop.Ksib,Veh(1).Prop.Ksib,Veh(1).Prop.Ksib];      % Suspension stiffness [N/m]
Veh(veh_num).Prop.cSi =  [1,1,2,2,2]*1e4;      % Suspension viscous damping [N*s/m]
Veh(veh_num).Prop.mSi =  [Veh(1).Prop.msif, Veh(1).Prop.msif, Veh(1).Prop.msib,Veh(1).Prop.msib, Veh(1).Prop.msib];      % Suspension mass [kg]
Veh(veh_num).Prop.afBi = [0,5];            % Distance from Body centre of gravity to location of front articulation [m]
Veh(veh_num).Prop.abBi = 4;                 % Distance from Body centre of gravity to location of back articulation [m]
Veh(veh_num).Prop.ei = [-1, 3.5, 1.2, 2.2, 3]  % Coordinate of axle (group) to centre of gravity of its corresponding body [m]
Veh(veh_num).Pos.vel = 12; %Veh(1).Pos.vel;                 %  Speed of vehicle [m/s] (negative value = vehicle moving from right to left)
Veh(veh_num).Pos.x0 = -100;                  % Initial position of vehicle (from left bridge support)



veh_num = veh_num + 1;
Veh(veh_num).Model.type = 'Truck_2';        % Text with name of model to use
Veh(veh_num).Prop.mBi =  Veh(2).Prop.mbd;   % Body masses [kg]
Veh(veh_num).Prop.IyBi = Veh(2).Prop.Iy;          % Body moments of inertia [kg*m2]
Veh(veh_num).Prop.kTi =  [Veh(2).Prop.Kt,Veh(2).Prop.Kt];  % Tyre stiffness [N/m]
Veh(veh_num).Prop.cTi =  [0,0]*1e4;       % Tyre viscous damping [N*s/m]
Veh(veh_num).Prop.kSi =  [Veh(2).Prop.Ks,Veh(2).Prop.Ks];      % Suspension stiffness [N/m]
Veh(veh_num).Prop.cSi =  [Veh(2).Prop.Cs,Veh(2).Prop.Cs];      % Suspension viscous damping [N*s/m]
Veh(veh_num).Prop.mSi =  [Veh(2).Prop.max,Veh(2).Prop.max];    % Suspension mass [kg]
Veh(veh_num).Prop.ei =   [-3, 0];         % Coordinate of axle (group) to centre of gravity of its corresponding body [m]
%Veh(veh_num).Pos.vel = -20;            % Speed of vehicle [m/s] (negative value = vehicle moving from right to left)
%Veh(veh_num).Pos.x0 = 110;             % Initial position of vehicle (from left bridge support)
Veh(veh_num).Pos.vel = 10%Veh(2).Pos.velb;

Veh(veh_num).Pos.x0 = Veh(2).Prop.S1;             % Initial pos

veh_num = veh_num + 1;
Veh(veh_num).Model.type = 'Truck_2';        % Text with name of model to use
Veh(veh_num).Prop.mBi =  Veh(2).Prop.mbd;   % Body masses [kg]
Veh(veh_num).Prop.IyBi = Veh(2).Prop.Iy;          % Body moments of inertia [kg*m2]
Veh(veh_num).Prop.kTi =  [Veh(2).Prop.Kt,Veh(2).Prop.Kt];  % Tyre stiffness [N/m]
Veh(veh_num).Prop.cTi =  [0,0]*1e4;       % Tyre viscous damping [N*s/m]
Veh(veh_num).Prop.kSi =  [Veh(2).Prop.Ks,Veh(2).Prop.Ks];      % Suspension stiffness [N/m]
Veh(veh_num).Prop.cSi =  [Veh(2).Prop.Cs,Veh(2).Prop.Cs];      % Suspension viscous damping [N*s/m]
Veh(veh_num).Prop.mSi =  [Veh(2).Prop.max,Veh(2).Prop.max];    % Suspension mass [kg]
Veh(veh_num).Prop.ei =   [-3, 0];         % Coordinate of axle (group) to centre of gravity of its corresponding body [m]
%Veh(veh_num).Pos.vel = -20;            % Speed of vehicle [m/s] (negative value = vehicle moving from right to left)
%Veh(veh_num).Pos.x0 = 110;             % Initial position of vehicle (from left bridge support)
Veh(veh_num).Pos.vel = -Veh(2).Pos.velb;   % Speed of vehicle [m/s] (negative value = vehicle moving from right to left)
Veh(veh_num).Pos.x0 = -Veh(2).Prop.S2;             % Initial pos

clear veh_num
% % ----------------- Calculation options and variables ---------------------
% -- Modal analysis --
Calc.Opt.veh_frq = 1;          % No calculation of Vehicle natural frequencies
Calc.Opt.beam_frq = 1;         % No calculation of Beam natural frequencies
Calc.Opt.beam_modes = 1;       % No calculation of Beam modes of vibration
% -- Solver step --
%Calc.Solver.min_t_steps_per_second = 1;        % Minimum time step per second
%Calc.Solver.min_Beam_modes_considered = 1;      % Number of beam modes to be considered for time step selection
Calc.Solver.max_accurate_frq =  150;              % [Hz]
%Calc.Solver.max_accurate_frq = 10;
%Calc.Solver.max_accurate_frq = 1000;
%Calc.Solver.max_accurate_frq = 10000;
% -- Newmark-Beta --
Calc.Solver.NewMark.damp = 1;     % Damped Newmark-Beta scheme used in Vehicle and Beam solvers
% -- Procedure to solve Interaction --
%Calc.Proc.name = 'FI';      % 1) Iterative procedure for whole time-history solution (Full iteration)
%Calc.Proc.name = 'SSI';     % 2) Step-by-Step iteration 
Calc.Proc.name = 'Coup';    % 3) Coupled system solution
% -- Full Iteration (FI) or Step-by-Step (SSI) options --
% Calc.Proc.Iter.max_num = 40;         % Maximum number of iterations (Default = 10)
% Calc.Proc.Iter.criteria = 1;         % Iteration criteria based on deformation under wheels (default)
% Calc.Proc.Iter.criteria = 2;         % Iteration criteria based on BM of whole beam
% Calc.Proc.Iter.tol = 1e-3;           % Tolerance for iteration stopping criteria (Default = Calc.Cte.tol)
% Calc.Proc.Iter.tol = 1e-20;
% -- Calculation options --
Calc.Opt.VBI = 1;                    % Disconnect VBI (Makes Iter.max_num = 0)
Calc.Opt.free_vib_seconds =0;        % [s] Additional free vibration to calculate bridge
%Calc.Opt.include_coriolis = 0;      % Include additional terms due to chain rule derivation (Default = 1)
% -- Display options --
Calc.Opt.verbose = 1;               % Display of comments
Calc.Opt.show_progress_every_s = 1; % Show progress every X seconds (Default = 2s)

% -------------------------------------------------------------------------
% % ------------------------- Plotting Options ------------------------------
%   Calc.Plot.P1_Beam_frq = 1;              % Distribution of beam frequencies
%   Calc.Plot.P2_Beam_modes = 5;           % First n modes of vibration (of Beam)
%   Calc.Plot.P3_VehPos = 1;                % Vehicle position (velocity and acceleration)
       Calc.Plot.P4_Profile = 1;               % Profiles and 1st derivative
 Calc.Plot.Profile_original = 1;         % Inside function B19
% Calc.Plot.P5_Beam_U = 1;                % Contourplot of Beam deformation
% Calc.Plot.P6_Beam_U_under_Veh = 1;      % Deformation under the vehicle
% %Calc.Plot.P7_Veh_U_iter = 1;            % Vehicle total displacement for each iteration (Only FI procedure)
% %Calc.Plot.P8_Veh_Urel_iter = 1;         % Vehicle relative displacement for each iteration (Only FI procedure)
% %Calc.Plot.P9_Beam_U_under_Veh_iter = 1; % Deformation under the vehicle for each iteration (Only FI procedure)
% %Calc.Plot.P10_diff_iter = 1;            % The difference between solutions (Iteration criteria) (Only FI procedure)
% %Calc.Plot.P11_Beam_U_static = 1;        % Calculates the Static Deformation of Beam (Due to Interaction force)
% Calc.Plot.P13_Interaction_Force = 1;    % Final interaction force
% %Calc.Plot.P14_Interaction_Force_iter = 1;  % Interaction force for each iteration (Only FI procedure)
% %Calc.Plot.P16_Beam_BM = 1;              % Contourplot of Beam BM
% %Calc.Plot.P17_Beam_BM_static = 1;       % Contourplot of Beam Static BM
% %Calc.Plot.P18_Beam_Shear = 1;           % Contourplot of Beam Shear
% %Calc.Plot.P19_Beam_Shear_static = 1;    % Contourplot of Beam Static Shear
% Calc.Plot.P20_Beam_MidSpan_BM = 1;      % Static and Dynamic BM at mid-span
% %Calc.Plot.P21_Beam_MidSpan_BM_iter = 1; % Static and Dynamic BM at mid-span for various iterations (Only FI procedure)
% %Calc.Plot.P22_SSI_num_iterations = 1;   % Number of iterations for each time step (Only for SSI procedure)
%  Calc.Plot.P23_vehicles_responses = 1;   % Vehicles DOF responses (Displacement, velocity and acceleration)
% %Calc.Plot.P24_vehicles_wheel_responses = 1; % Responses at the wheels of the vehicle
%  Calc.Plot.P25_vehicles_under_responses = 1; % Responses under the wheel (Bridge response at the contact point)
%    Calc.Plot.P26_PSD_Veh_A = 1;            % PSD of vehicle accelerations
%    Calc.Plot.P27_Veh_A = 1;                % Time histories of vehicle accelerations

% -------------------------------------------------------------------------
% ------------------------- Calculations ----------------------------------

A02_VBI_calculations;

% -------------------------------------------------------------------------
% ---------------------------- Plotting -----------------------------------

B44_Plots;

% -------------------------------------------------------------------------
% ---------------------------- VMD_TEST -----------------------------------
% tic
% disp('Runing VMD')
% VMD_test;
% toc
% % -------------------------------------------------------------------------                                 
% % ---------------------------- VMD_TEST -----------------------------------
% tic
% disp('Running EWT')
% Test_EWT1D;
% toc

% tic
% VBI_Analysis;
% toc
% -------------------------------------------------------------------------
% Removing functions folder path
%rmpath 'Functions'

% ---- End of script ----    