% This is the same as "myclearbut" script with the only difference that
% the variable "do_not_delete" is not deleted at the end.

% This script is used to delete all variables in the workspace except for
% those defined in the cell
%   do_not_delete = {'VariableName1','VariableName2',...}

% Note: This is not a function

% Note: This script creates and deletes several variables
%       Check that these variable names are not used in the main script
%             CurrentVariablesInWorspaceS 
%             do_not_delete2 
%             index_k 
%             index_k2 
%             index_k3

clear CurrentVariablesInWorspaceS
CurrentVariablesInWorspaceS = whos;

% Adding "do_not_delete" to the list
do_not_delete2 = do_not_delete;
do_not_delete2{end+1} = 'do_not_delete';

% Loop: Variable names in currect workspace
for index_k = 1:size(CurrentVariablesInWorspaceS,1)
    
    index_k3 = 0;
    
    % Checking if variable name is in the list "do_not_delete"
    for index_k2 = 1:size(do_not_delete2,2)
        if strcmp(CurrentVariablesInWorspaceS(index_k).name,do_not_delete2{index_k2})
            index_k3 = 1;
        end % if ~strcmp(CurrentVariablesInWorspaceS(index_k).name,do_not_delete2{index_k2})
    end % for index_k2 = 1:size(do_not_delete2,2)
    
    % Clear variable if not in the list 
    if index_k3 == 0
        clear(CurrentVariablesInWorspaceS(index_k).name);
    end % if index_k3 == 0
    
end % for index_k = 1:size(CurrentVariablesInWorspaceS,1)

clear CurrentVariablesInWorspaceS do_not_delete2 index_k index_k2 index_k3

% ---- End of script ----