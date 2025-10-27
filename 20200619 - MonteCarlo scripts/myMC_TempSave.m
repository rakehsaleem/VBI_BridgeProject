% Note: this is not a function

% Temporary save of results

% % -------------------------------------------------------------------------
% % ----- Inputs ----
% MC = Structure with Monte Carlo information, at least:
%   .Save.path_name = folder path to save results
%   .Save.file_name = name of the file name to be saved
%   .TempSave.num_variables_2_save = number of variables to save
%   .TempSave.variables_2_save = Cell array listing the names of the variables to save
%   .TempSave.every = Time in minutes when a temporary save should be generated
%   .TempSave.next_save_t = Time when the next temporary save should be done
% % ----- Outputs ----
% % -------------------------------------------------------------------------

if MC.Calc.PCtime.value/60 > MC.TempSave.next_save_t
    
    % Generating saving command from list of names to save
    mytext = '''';
    for variable_num = 1:MC.TempSave.num_variables_2_save
        mytext = [mytext,MC.TempSave.variables_2_save{variable_num},''','''];
    end % for variable_num = 1:MC.TempSave.num_variables_2_save
    mytext = ['''',MC.Save.path_name,'TEMP_',MC.Save.file_name,'.mat'',',mytext(1:end-2)];
    mytext = ['save(',mytext,')'];
    
    % Saving
    eval(mytext);
    
    % Updating variables
    MC.TempSave.next_save_t = MC.TempSave.next_save_t + MC.TempSave.every;
    
end % if MC.Calc.PCtime.value/60 > MC.TempSave.next_save_t

% ---- End of script ----