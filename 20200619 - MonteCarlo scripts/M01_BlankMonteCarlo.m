


% ---- Monte Carlo script ----

% This is a blank Monte Carlo script

% -------------------------------------------------------------------------
close all;
clear all;
clear; clc; rng('shuffle');


% -------------------------------------------------------------------------
% ----------------------- Monte Carlo Options -----------------------------

% ---- Size of Monte Carlo analysis ----
% Select only one of the following 3 options :



% Option 1 - Number of runs
%MC.Opt.num_runs = 1;
MC.Opt.num_runs = 1000;
%MC.Opt.num_runs = 50;

% Option 2 - Simulation time [Days Hours Minutes Seconds]
%MC.Opt.simulation_time = [0 0 0 5];
%MC.Opt.simulation_time = [0 0 25 0];
%MC.Opt.simulation_time = [0 2 0 0]; 
%MC.Opt.simulation_time = [15 0 0 0];

% Option 3 - Finish date [Year Month Day Hour Minute Second]
%MC.Opt.finish_date = '2020 02 06 10 49 30';

% ---- Results Save information ----





MC.Save.file_name = 'Test_Damage';
%MC.Save.file_name = '13102021_Five_Axle_profile_damage_Fixed_Speed_125_10_testing';

MC.Save.file_name = 'Final_Paper_Long_Bridge_Traffic_Damage_SUPPORT2_30_New';
MC.Save.path_name = ['Results\',mfilename,'\'];
MC.Save.single_file = 0;    % Saves all results into a signel file (Default = 0)

MC.TempSave.every = 30;     % Temporary save every XX minutes (Only if MC.Save.single_file == 1)
MC.TempSave.variables_2_save = {'MC','MCSol'}; % Cell with list of variables to save

% ---- Monte Carlo Variability ---- 
% Available distributions
% 'uni' = Uniform distribution (only integers)  Inputs = [min, max]
% 'un2' = Uniform distribution                  Inputs = [min, max]
% 'nor' = Normal distribution (with limits)     Inputs = [min, max, mean, std]
% 'nol' = Normal distribution (no limits)       Inputs = [mean, std]
% 'arr' = Random values from array              Inputs = [Array of values]
%% QUARTER CAR
% %mass
% MC.Var.Veh.Prop.mb = myMC_varDef([1200,2400,1600,100],'nor','Vehicle mass (m/s)');
% MC.Var.Veh.Prop.msi = myMC_varDef([600,1200,800,50],'nor','Vehicle suspension mass kg');
% %suspension
% MC.Var.Veh.Prop.kti = myMC_varDef([1.5*1e6,2*1e6,1.75*1e6,0.1e6],'nor','Suspension Properties(Nm/s)');
% 
% MC.Var.Veh.Pos.vel = myMC_varDef([10,16,13,1.5],'nor','Vehicle velocity (m/s)');
%% Two Axle Truck
%%Velocity
MC.Var.Veh.Pos.velr = myMC_varDef([11*0.44704,15*0.44704,13*0.44704,1*0.44704],'nor','Vehicle velocity (m/s)'); % 1 mph = 0.44704 m/s
MC.Var.Veh.Pos.velb = myMC_varDef([11*0.44704,15*0.44704,13*0.44704,1*0.44704],'nor','Vehicle velocity (m/s)');
% Body
MC.Var.Veh.Prop.mb = myMC_varDef([10000,11000,10500,20],'nor','Vehicle mass (Kg)');
% Axle 
MC.Var.Veh.Prop.ma = myMC_varDef([700,800,750,10],'nor','Vehicle suspension mass kg');
% Suspension System
MC.Var.Veh.Prop.ksb = myMC_varDef([4*1e6,8*1e6,6*1e6,0.2e6],'nor','Suspension Properties(Nm/s)');
% Tyre Suspension System
MC.Var.Veh.Prop.ktt = myMC_varDef([1.5*1e6,2*1e6,1.75*1e6,500],'nor','Suspension Properties(Nm/s)');
%
MC.Var.Veh.Prop.Csb = myMC_varDef([0.9*1e4,1.1*1e4,1*1e4,100],'nor','Suspension Properties(Nm/s)');
% Momet of inertia
MC.Var.Veh.Prop.I = myMC_varDef([52000,55000,53651,100],'nor','Momet of inertia');
% Axle dimension
MC.Var.Veh.Prop.ax = myMC_varDef([-2 -9],'uni','Front Body Dim');

% Starting Position
MC.Var.Veh.Prop.x0_1 = myMC_varDef([-90,-80,-85,1],'nor','starting position 1');
MC.Var.Veh.Prop.x0_2 = myMC_varDef([-220,-180,-190,1],'nor','starting position 2');

%% Five Axle Vehicle
% % Vehicle velocity
% %MC.Var.Veh.Pos.vel = myMC_varDef([50,100,75,10]/3.6,'nor','Vehicle velocity (m/s)');
% MC.Var.Veh.Pos.vel = myMC_varDef([10,16,13,1],'nor','Vehicle velocity (m/s)');
% %MC.Var.Veh.Pos.vel = myMC_varDef([8,12,10,1],'nor','Vehicle velocity (m/s)');
% 
% 
% % Vehicle body mass
% MC.Var.Veh.Prop.mbf = myMC_varDef([2800,3800,3300,100],'nor','Vehicle mass Front');
% MC.Var.Veh.Prop.mbb = myMC_varDef([10000,40000,25000,4500],'nor','Vehicle mass Back');
% % Vehicle Axle mass
% MC.Var.Veh.Prop.msif = myMC_varDef([300,900,600,80],'nor','Vehicle suspension mass kg');
% MC.Var.Veh.Prop.msib = myMC_varDef([800,1200,1000,80],'nor','Vehicle suspension mass Back kg');
% 
% % Suspension System
% MC.Var.Veh.Prop.ksif = myMC_varDef([4*1e6,8*1e6,6*1e6,0.2e6],'nor','Suspension Properties(Nm/s)');
% MC.Var.Veh.Prop.ksib = myMC_varDef([8*1e6,12*1e6,10*1e6,0.2e6],'nor','Suspension Properties(Nm/s)');
% 
% % Tyre Suspension System
% MC.Var.Veh.Prop.ktif = myMC_varDef([1.25*1e6,2.25*1e6,1.75*1e6,0.2e6],'nor','Suspension Properties(Nm/s)');
% MC.Var.Veh.Prop.ktib = myMC_varDef([2.75*1e6,4.75*1e6,3.5*1e6,0.2e6],'nor','Suspension Properties(Nm/s)');
% % momet of inertia
% 
% MC.Var.Veh.Prop.Iyf = myMC_varDef([4250,5500,4875,25],'nor','Momet of inertia');
% MC.Var.Veh.Prop.Iyb = myMC_varDef([112000,1350000,123000,2000],'nor','Momet of inertia');
% 
% MC.Var.Veh.Prop.afBi = myMC_varDef([5,6,5.5,0.1],'nor','Front_Body_Dim');
% MC.Var.Veh.Prop.abBi = myMC_varDef([4,4.5,4.25,0.02],'nor','Back_Body_Dim');
%% Fleet Vehicles
%%Vehicle velocity
% MC.Var.Veh.Pos.vel = myMC_varDef([100,140,120,10]/3.6,'nor','Vehicle velocity (m/s)');
% MC.Var.Veh.Pos.vel = myMC_varDef([60,100,80,10]/3.6,'nor','Vehicle velocity (m/s)');
% MC.Var.Veh.Pos.vel = myMC_varDef([10,16,13,1],'nor','Vehicle velocity (m/s)');

MC.Var.Veh.Pos.vel = myMC_varDef([10,15.5,13,1],'nor','Vehicle velocity (m/s)');

% MC.Var.Veh.Pos.vel = myMC_varDef([8,12,10,1],'nor','Vehicle velocity (m/s)');



%Vehicle body mass
MC.Var.Veh.Prop.mbf = myMC_varDef([3100,3300,3200,20],'nor','Vehicle mass Front');
MC.Var.Veh.Prop.mbb = myMC_varDef([15000,25000,20000,1000],'nor','Vehicle mass Back');
% Vehicle Axle mass

MC.Var.Veh.Prop.msif = myMC_varDef([540,560,550,1],'nor','Vehicle suspension mass kg');
MC.Var.Veh.Prop.msib = myMC_varDef([990,1010,1000,1],'nor','Vehicle suspension mass kg');

% Suspension System
MC.Var.Veh.Prop.ksif = myMC_varDef([4*1e6,8*1e6,6*1e6,0.02e6],'nor','Suspension Properties(Nm/s)');
MC.Var.Veh.Prop.ksib = myMC_varDef([8*1e6,12*1e6,10*1e6,0.2e6],'nor','Suspension Properties(Nm/s)');

% Tyre Suspension System
MC.Var.Veh.Prop.ktif = myMC_varDef([1.715e6,1.785e6,1.75*1e6,500],'nor','Suspension Properties(Nm/s)');
MC.Var.Veh.Prop.ktib = myMC_varDef([3430000,3570000,3.5*1e6,1000],'nor','Suspension Properties(Nm/s)');
% momet of inertia

MC.Var.Veh.Prop.Iyf = myMC_varDef([4250,5500,4875,25],'nor','Momet of inertia');
MC.Var.Veh.Prop.Iyb = myMC_varDef([112000,1350000,123000,2000],'nor','Momet of inertia');
% % %%
% % ---- MC calculations option ----
 MC.Opt.checkRandDist_on = 1;        % Display histograms of all random variables

% -------------------------------------------------------------------------
% --------------------- Monte Carlo calculations --------------------------



[MC] = myMC_auxiliary_Calcs(MC);

while MC.Calc.continue == 1

    % MC analysis time calculations (and other)
    MC = myMC_Time(MC);
    
    % Clearing workspace
    do_not_delete = {'MC','MCSol'};
    myclearbut2;
  
    % Seeding the random simulator
    rng(239250042+ MC.Calc.counter);
     %MC.Calc.org_seed
    % **************************** MISSING ************************************
    %239250042 five axle
    % Include here your input defintions and calculations

    % Use randomly generated values as follows:
    %%
    Veh(1).Pos.vel = myrandS(MC.Var.Veh(1).Pos.vel);          % Vehicles velocity [m/s]
    
    %Mass
%     Veh(1).Prop.mb = myrandS(MC.Var.Veh(1).Prop.mb);
%     Veh(1).Prop.msi = myrandS(MC.Var.Veh(1).Prop.msi);
    
    % Suspension
%     Veh(1).Prop.Kti= myrandS(MC.Var.Veh(1).Prop.kti);
%     Veh(1).Pos.vel = myrandS(MC.Var.Veh(1).Pos.vel);         
    
    Veh(1).Prop.mbf = myrandS(MC.Var.Veh(1).Prop.mbf);
    Veh(1).Prop.mbb = myrandS(MC.Var.Veh(1).Prop.mbb);
    
    Veh(1).Prop.msif= myrandS(MC.Var.Veh(1).Prop.msif);
    Veh(1).Prop.msib= myrandS(MC.Var.Veh(1).Prop.msib);
   
    Veh(1).Prop.Ksif= myrandS(MC.Var.Veh(1).Prop.ksif);
    Veh(1).Prop.Ksib= myrandS(MC.Var.Veh(1).Prop.ksib);
    
    Veh(1).Prop.Ktif= myrandS(MC.Var.Veh(1).Prop.ktif);
    Veh(1).Prop.Ktib= myrandS(MC.Var.Veh(1).Prop.ktib);
    
    Veh(1).Prop.Iyf= myrandS(MC.Var.Veh(1).Prop.Iyf);
    Veh(1).Prop.Iyb= myrandS(MC.Var.Veh(1).Prop.Iyb);
    
    
%    Veh(1).Prop.Lfa= myrandS(MC.Var.Veh(1).Prop.afBi);
 %   Veh(1).Prop.Lba= myrandS(MC.Var.Veh(1).Prop.abBi);
%%  Two Axle Truck
%     Veh(2).Pos.velr = myrandS(MC.Var.Veh(1).Pos.velr);         
%     Veh(2).Pos.velb = myrandS(MC.Var.Veh(1).Pos.velb); 
%     
%     Veh(2).Prop.mbd = myrandS(MC.Var.Veh(1).Prop.mb);
%     Veh(2).Prop.max= myrandS(MC.Var.Veh(1).Prop.ma);
%     
%     
%     Veh(2).Prop.Ks= myrandS(MC.Var.Veh(1).Prop.ksb);
%     Veh(2).Prop.Kt= myrandS(MC.Var.Veh(1).Prop.ktt);
%     
%     
% 
%     Veh(2).Prop.Cs= myrandS(MC.Var.Veh(1).Prop.Csb);
%     
%     Veh(2).Prop.Iy= myrandS(MC.Var.Veh(1).Prop.I);
%     
%     Veh(2).Prop.Spa= myrandS(MC.Var.Veh(1).Prop.ax);
%     
%     Veh(2).Prop.S1= myrandS(MC.Var.Veh(1).Prop.x0_1);
%     Veh(2).Prop.S2= myrandS(MC.Var.Veh(1).Prop.x0_2);
%%




    % *************************************************************************
    A01_VBI_input_and_run
   % A01_VBI_input_and_run_Long_Bridge
    % -------------------------------------------------------------------------
    % ----------------------- Variables to Save -------------------------------

    
    % Monte Carlo results to save
%     Run.Veh = Veh;
%     Run.MC.Calc.counter = MC.Calc.counter;
%     Run.Veh(1).Solver=Calc.Solver;
%     Run.Veh(1).time=Calc.Solver.t;
%     Run.Veh(1).beam=Calc.Solver.t_beam;
%     Run.Sol.Beam = Sol.Beam;
%     Run.Sol.Veh = Sol.Veh(1);
%     Run.Beam.f=Beam.Modal.f;
    
    Run.Veh = Veh;
%     Run.MC.Calc.counter = MC.Calc.counter;
%     Run.Veh(1).Solver=Calc.Solver;
%     Run.Veh(1).time=Calc.Solver.t;
%     Run.Veh(1).beam=Calc.Solver.t_beam;
    Run.Sol.Beam = Sol.Beam;
    Run.Sol.Veh = Sol.Veh(1);
    Run.Beam.f=Beam.Modal.f;    
    % Saving
    
    myMC_Save;
    
    % Updating counter
    [MC] = myMC_counter_and_pause(MC);
    
end % while MC.Calc.continue == 1

myclearbut;
disp('All calculations finished sucessfully');

% -------------------------------------------------------------------------
% ------------------------- Saving results --------------------------------

myMC_Save;

% ---- End of script ----