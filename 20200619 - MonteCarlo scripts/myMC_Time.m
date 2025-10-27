function [MC] = myMC_Time(MC)

% Based on the MonteCarlo input parameters calculates:
%   Total calculation time needed for th MC simulation
%   Initialize auxiliary variables
%   Display simulation number

% -------------------------------------------------------------------------
% ---- Input ----
% MC = Structure with MonteCarlo variables, including at least:
% ---- Output ----
% MC = Addition of new fields to the structure MC
% -------------------------------------------------------------------------

if ~isfield(MC.Calc,'counter')
    
    % Checking input
    aux1 = sum([isfield(MC.Opt,'num_runs'),isfield(MC.Opt,'simulation_time'),isfield(MC.Opt,'finish_date')]);
    if aux1 > 1
        disp('Number of Runs / Simulation time / Finish date cannot be defined simultaneously');
        error('Number of Runs / Simulation time / Finish date cannot be defined simultaneously');
    elseif aux1 == 0
        disp('How many runs should the simulation have?')
        disp('How long should the simulation run?')
        disp('When should the simulation stop?')
        error('Stop condition needed, either number of runs, length of simulation or finish date');
    end % if aux1 > 1

    % Adding Year and Month to simulation time
    if isfield(MC.Opt,'simulation_time')
        MC.Calc.PCtime.end = [0,0,MC.Opt.simulation_time];
    end % if isfield(MC.Opt,'simulation_time')

    % Calculation of PCtime.end from MC.Opt.finish_date
    if isfield(MC.Opt,'finish_date')
        MC.Calc.PCtime.end = datenum(MC.Opt.finish_date,'yyyy mm dd HH MM SS') - now;
        MC.Calc.PCtime.end = datevec(MC.Calc.PCtime.end);
        if MC.Calc.PCtime.end(2)>0
            MC.Calc.PCtime.end = MC.Calc.PCtime.end - [0,1,0,0,0,0];
        end % if MC.Calc.PCtime.end(2)>0
        MC.Opt.simulation_time = MC.Calc.PCtime.end;
        
        % Checking for negative values
        if any(MC.Calc.PCtime.end<0)
            disp('Finished date set in the past!');
            error('Finished date set in the past!');
        end % if any(MC.Calc.PCtime.end<0)
    end % if isfield(MC.Opt,'finish_date')

    % Default values
    if ~isfield(MC.Opt,'disp_every_t')
        MC.Opt.disp_every_t = 2;
    end % if ~isfield(MC.Opt,'disp_every_t')
    
    % Starting the clock
    MC.Calc.counter = 1;
    MC.Calc.PCtime.start = clock;
    MC.Calc.last_display_time = MC.Opt.disp_every_t;
    if isfield(MC.Opt,'num_runs')
        disp(['MonteCarlo simulation started (',num2str(MC.Opt.num_runs),' runs)'])
    elseif isfield(MC.Opt,'simulation_time')
        aux1 = []; aux2 = 0;
        aux_text = {'years','months','days','hours','minutes','seconds'};
        for i = 1:length(MC.Calc.PCtime.end)
            if MC.Calc.PCtime.end(i) > 0
                aux2 = aux2 + 1;
            end % if MC.Calc.PCtime.end(i) > 1
            if aux2 > 0
                aux1 = [aux1,num2str(MC.Calc.PCtime.end(i)),' ',aux_text{i},' '];
            end % if aux2 > 0
        end % for i = 1:length(MC.Calc.PCtime.end)
        disp(['MonteCarlo simulation started (',aux1(1:end-1),')']);
        disp(['Simulation should finish at: ',datestr(MC.Calc.PCtime.start + MC.Calc.PCtime.end)]);
    end % if isfield(MC.Opt,'num_runs')
    if isfield(MC,'save_Project_results')
        disp(['Results will be saved in folder MC_Results\Results_',MC.Project_Name]);
    end % if isfield(MC,'save_Project_results')
    
end % if ~isfield(MC.Calc,'counter')

% PCtime of running Simulation
MC.Calc.PCtime.value = etime(clock,MC.Calc.PCtime.start);

% Display simulation number
if MC.Calc.PCtime.value > MC.Calc.last_display_time
    disp(['Run ',num2str(MC.Calc.counter)]);
    MC.Calc.last_display_time = MC.Calc.last_display_time + MC.Opt.disp_every_t;
end % if MC.Calc.PCtime.value > MC.Calc.last_display_time

% ---- End of function ----