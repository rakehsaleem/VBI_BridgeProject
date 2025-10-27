function [ ] = checkRandDist(input_structure,varargin)

% Checks the defined random distributions graphically
% It is possible to study 
%   single varaibles variability
%   several variables variability

% It is possible to terminate the execution by typing letter 'q' when promped
% on the command window.

% -------------------------------------------------------------------------
% ---- Inputs ----
% input_structure = Structure that contains at least the fields:
%   Option 1: Single variable variability
%   .dist = Distribution Considered:   
%         'uni' = Uniform distribution (only scalars)
%         'un2' = Uniform distribution
%         'nor' = Normal distribution (with limits)
%         'nol' = Normal distribution (no limits)
%         'arr' = Random values from array
%   .var = array that differs for each distribution
%       % -- Inputs for Uniform distribution --
%       [mini,maxi] = [Mininmum value, Maximum value]
%       % -- Inputs for Normal distribution with limits --
%       [mini, maxi, mean_norm, std_norm] = 
%           Mininmum value
%           Maximum value
%           Mean value of Normal distribution
%           Standard deviation of Normal distribution
%       % -- Inputs for Normal distribution No limits --
%       [mean_norm, std_norm] = 
%           Mean value of Normal distribution
%           Standard deviation of Normal distribution
%       % -- Inputs for Array random value --
%       array_in = Array of values
%   Option 2: Several variables variability
%   One substructure containing the same information as in Option 1 for every
%       variable to be considered
% ---- Optional inputs ----
% N = Number of random samples for each histogram (Default = 10000)
% ---- Outputs ----
% A histogram
% -------------------------------------------------------------------------

% Calculation options
if nargin == 1
    N = 10000;   % Number of samples
elseif nargin == 2
    N = varargin{1};
end % if nargin == 1

% ---- Input only one variability ----
if and(isfield(input_structure,'var'),isfield(input_structure,'dist'))

    input_structure.N = N;
    histplot(input_structure);
    
% ---- Multiple variability ----
else
    
    % Definition of all names of fields and subfields (Automatically)
    counter = 1;
    auxS = input_structure;
    names1 = fieldnames(auxS);
    % Level 1
    for field1 = 1:size(names1,1)
    auxS = input_structure.(names1{field1});
        if ~isfield(auxS,'var')
            names2 = fieldnames(auxS);
            % Level 2
            for field2 = 1:size(names2,1)
                auxS = input_structure.(names1{field1}).(names2{field2});
                if ~isfield(auxS,'var')
                    names3 = fieldnames(auxS);
                    % Level 3
                    for field3 = 1:size(names3,1)
                        auxS = input_structure.(names1{field1}).(names2{field2}).(names3{field3});
                        if ~isfield(auxS,'var')
                            names4 = fieldnames(auxS);
                            % Level 4
                            for field4 = 1:size(names4,1)
                                auxS = input_structure.(names1{field1}).(names2{field2}).(names3{field3}).(names4{field4});
                                if ~isfield(auxS,'var')
                                    names5 = fieldnames(auxS);
                                    % Level 5
                                    for field5 = 1:size(names5,1)
                                        auxS = input_structure.(names1{field1}).(names2{field2}).(names3{field3}).(names4{field4}).(names5{field5});
                                        if ~isfield(auxS,'var')
                                            names6 = fieldnames(auxS);
                                            % Level 6
                                            for field6 = 1:size(names6,1)
                                                auxS = input_structure.(names1{field1}).(names2{field2}).(names3{field3}).(names4{field4}).(names5{field5}).(names6{field6});
                                                if ~isfield(auxS,'var')
                                                    names7 = fieldnames(auxS);
                                                    % Level 7
                                                    for field7 = 1:size(names7,1)
                                                        auxS = input_structure.(names1{field1}).(names2{field2}).(names3{field3}).(names4{field4}).(names5{field5}).(names6{field6}).(names7{field7});
                                                        if ~isfield(auxS,'var')
                                                            names8 = fieldnames(auxS);
                                                            load_names{counter,:} = {names1{field1},names2{field2},names3{field3},names4{field4},names5{field5},names6{field6},names7{field7},names8{field8}};
                                                            counter = counter + 1;
                                                        end % if ~isfield(auxS,'var')
                                                    end % for field7 = 1:size(names7,1)
                                                else
                                                    load_names{counter,:} = {names1{field1},names2{field2},names3{field3},names4{field4},names5{field5},names6{field6}};
                                                    counter = counter + 1;
                                                end % if ~isfield(auxS,'var')
                                            end % for field6 = 1:size(names6,1)
                                        else
                                            load_names{counter,:} = {names1{field1},names2{field2},names3{field3},names4{field4},names5{field5}};
                                            counter = counter + 1;
                                        end % if ~isfield(auxS,'var')
                                    end % for field5 = 1:size(names5,1)
                                else
                                    load_names{counter,:} = {names1{field1},names2{field2},names3{field3},names4{field4}};
                                    counter = counter + 1;
                                end % if ~isfield(auxS,'var')
                            end % for field4 = 1:size(names4,1)
                        else
                            load_names{counter,:} = {names1{field1},names2{field2},names3{field3}};
                            counter = counter + 1;
                        end % if ~isfield(auxS,'var')
                    end % for field3 = 1:size(names3,1)
                else
                    load_names{counter,:} = {names1{field1},names2{field2}};
                    counter = counter + 1;
                end % if ~isfield(auxS,'var')
            end % for field2 = 1:size(names2,1)
        else
            load_names{counter,:} = {names1{field1}};
            counter = counter + 1;
        end % if ~isfield(auxS,'var')
    end % for field1 = 1:size(names1,1)

    clear field1 field2 field3 field4 field5 names1 names2 names3 names4 names5 counter

    figure; myFigMaximize;
    counter = 0;
    sub_num_rows = ceil(sqrt(size(load_names,1)));
    sub_num_cols = ceil(size(load_names,1)/sub_num_rows);
    
    % Variable loop
    for k = 1:size(load_names,1)

        num_levels = size(load_names{k,:},2);

        % Definition of new values
        if num_levels == 1
            aux_structure = input_structure.(load_names{k}{1});
        elseif num_levels == 2
            aux_structure = input_structure.(load_names{k}{1}).(load_names{k}{2});
        elseif num_levels == 3
            aux_structure = input_structure.(load_names{k}{1}).(load_names{k}{2}).(load_names{k}{3});
        elseif num_levels == 4
            aux_structure = input_structure.(load_names{k}{1}).(load_names{k}{2}).(load_names{k}{3}).(load_names{k}{4});
        elseif num_levels == 5
            aux_structure = input_structure.(load_names{k}{1}).(load_names{k}{2}).(load_names{k}{3}).(load_names{k}{4}).(load_names{k}{5});
        elseif num_levels == 6
            aux_structure = input_structure.(load_names{k}{1}).(load_names{k}{2}).(load_names{k}{3}).(load_names{k}{4}).(load_names{k}{5}).(load_names{k}{6});
        end % if num_levels == 1

        counter = counter + 1;
        %subplot(sub_num_rows,sub_num_cols,counter);
        mysubplot(sub_num_rows,sub_num_cols,counter,[1,0.2,1,0.2]*0.2);
        
        % Plots histogram
        aux_structure.N = N;
        histplot(aux_structure); axis tight;
        
    end % for k = 1:size(names,1)
    
    % User input
    user_input = input('Press any key to continue with the MC simulation (or "q" to quit) ','s');

    % Closing figure
    close;

    % Finish execution
    if user_input == 'q'
        error;
    end % if user_input == 'q'
    
end % if and(isfield(input_structure,'var'),isfield(input_structure,'dist'))

% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
function [] = histplot(input_structure)

    % Initialize variables
    Data = zeros(1,input_structure.N);

    % Calculations
    for n = 1:input_structure.N
        Data(n) = myrandS(input_structure);
    end % for n = 1:N

    % Plotting resutls
    hist(Data,40);

    % X Label
    if isfield(input_structure,'text')
        xlabel(input_structure.text)
    end % if isfield(my_structure,'text');
    
    if ischar(input_structure.var(1))
        Data = sort(Data);
        xtick = Data([1,diff(Data)]>0);
        set(gca,'XTick',xtick);
        for ind = 1:length(input_structure.var)
            xtick_txt{ind} = input_structure.var(ind);
        end % for ind = 1:length(input_structure.var)
        set(gca,'XTickLabel',xtick_txt)
    end % if ischar(input_structure.var(1))
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

% ---- End of function ----