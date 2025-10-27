% Note: This is not a function

% Saving MC results
% This script can be used within the while loop of the MC analysis. In that
%   case, depending on the selected option, it saves the results in separate
%   files or gathers the current run results into one single file. If necessary
%   it also performs the temporary saving of the results
% If used outside the while loop, then it save the results (when single file
%   option is defined), checks that the file does not exist already and removes
%   any eventual temporary file.

% Used inside while loop
if MC.Calc.continue == 1
    
    % Saving results for each run in a separate file
    if MC.Save.single_file == 0
    
        % File name generation
        datenow_txt = datestr(now,'yyyymmddHHMMSSFFF');
        file_name = [datenow_txt(1:8),'_',datenow_txt(9:end),'_',MC.Save.file_name];

        % Saving MC information (and Calc structure)
        if MC.Calc.counter == 1
            file_name_MC = [datenow_txt(1:8),'_',datenow_txt(9:end),'_0_MC_file'];
            save([MC.Save.path_name,file_name_MC],'MC','Calc');
            disp(['Results saved as: ',MC.Save.path_name,file_name_MC,'.mat']);
            clear file_name_MC
        end % if MC.Calc.counter == 1
        
        % Saving
        save([MC.Save.path_name,file_name],'Run');
        disp(['Results saved as: ',MC.Save.path_name,file_name,'.mat']);
        
        clear datenow_txt file_name
    
    % Saving all MC result in on single file
    elseif MC.Save.single_file == 1
        
        % Gathering into 1 file
        MCSol.Run(MC.Calc.counter) = Run;
    
        % Temporary save
        myMC_TempSave;
    
    end % if MC.Save.single_file == 1

% Used when all simulations are done (outside while loop)
elseif MC.Calc.continue == 0
    
    if MC.Save.single_file == 1

        disp('Saving results ...');

        % Checking if already exists
        if exist([MC.Save.path_name,MC.Save.file_name,'.mat'],'file')
            disp('File already exists with this name:');
            disp([MC.Save.path_name,MC.Save.file_name,'.mat']);
            user_in = input('Do you want to REPLACE it? Type then "Y"  ','s');
            if user_in == 'Y'; k_save = 1; else k_save = 0; end
        else
            k_save = 1;
        end % if exist([MC.Save.path_name,MC.Save.file_name,'.mat'],'file')
        clear user_in

        % Saving results
        if k_save == 1
            % Generating saving command from list of names to save
            mytext = '''';
            for variable_num = 1:MC.TempSave.num_variables_2_save
                mytext = [mytext,MC.TempSave.variables_2_save{variable_num},''','''];
            end % for variable_num = 1:MC.TempSave.num_variables_2_save
            mytext = ['''',MC.Save.path_name,'TEMP_',MC.Save.file_name,'.mat'',',mytext(1:end-2)];
            mytext = ['save(',mytext,')'];
            % Saving
            eval(mytext);
            save([MC.Save.path_name,MC.Save.file_name],'MC','MCSol','-v7.3');
            disp(['Results saved as: ',MC.Save.path_name,MC.Save.file_name,'.mat']);
        else
            disp('Results NOT saved');
        end % if k_save == 1
        clear k_save mytext variable_num

        % Deleting Temporary save
        if exist([MC.Save.path_name,'TEMP_',MC.Save.file_name,'.mat'],'file')
            delete([MC.Save.path_name,'TEMP_',MC.Save.file_name,'.mat']);
        end % if exist([MC.Save.path_name,'TEMP_',MC.Save.file_name,'.mat'],'file')

        fprintf('\b'); disp('  DONE');

    end % if MC.Save.single_file == 1
end % if MC.Calc.continue == 1

% ---- End of script ----