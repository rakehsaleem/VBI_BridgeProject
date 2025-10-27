function [MC] = myMC_auxiliary_Calcs(MC)

% Auxiliary calculations before starting a Monte Carlo analysis, including:
%   - Graphical check of variability of the random variables
%   - Creation (if necessary) of destination folder for saving the results
%   - Initial seed for the random number generation
%   - Other auxiliary variables

% % -------------------------------------------------------------------------
% % ----- Inputs ----
% MC = Structure with MonteCarlo variables, including at least:
%   .Opt.checkRandDist_on = Flag to activate the graphical check of the random
%       variables variability
%   .Save.path_name = String with the path of the folder where results should
%       be saved
% Optional fields
%   .Save.single_file = Logical file indicating if all MC analysis results should
%       be saved (0) as separate files for each run or (1) into one single file
%       (Deafult = 0)
%   .TempSave = Substructure with information about temporary saving of results
%       This is only applicable if MC.Save.single_file = 1
%       .variables_2_save = Cell array listing the names of the variables to save
%       .every = Time in minutes when a temporary save should be generated
% ---- Output ----
% MC = Addition of new fields to the structure MC
%   .Calc.org_seed = Initial random generation seed defined before the first MC run
%   .Calc.continue = Logical flag that indicates if the MC while loop needs to continue
%   .Save.single_file = If not defined before, the default value is provided
%   .TempSave = Substructure with information about temporary saving of results
%       This is only applicable if MC.Save.single_file = 1
%       .num_variables_2_save = number of variables to save
%       .next_save_t = Time when the next temporary save should be done
% % -------------------------------------------------------------------------

% Display of random variables histograms
if myIsfield(MC,{'Opt',1,'checkRandDist_on'},1)
    checkRandDist(MC.Var,10000);
end % if myIsfield(MC,{'Opt',1,'checkRandDist_on'},1)

% Checking existence and creating folder to save results
aux1 = dir(MC.Save.path_name);
if isempty(aux1)
    mkdir(MC.Save.path_name);
end % if length(aux1) == 0

% Random generation seed
MC.Calc.org_seed = int32(round(rand*10^9));

% Auxiliary variables
MC.Calc.continue = 1; 

% Default values
if ~myIsfield(MC,{'Save',1,'single_file'})
    MC.Save.single_file = 0;
end % if ~myIsfield(MC,{'Save',1,'single_file'})

if MC.Save.single_file == 1
    MC.TempSave.num_variables_2_save = length(MC.TempSave.variables_2_save);
    MC.TempSave.next_save_t = MC.TempSave.every;
end % if MC.Save.single_file == 1

% ---- End of script ----