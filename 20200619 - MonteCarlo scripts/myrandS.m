function [randvalue] = myrandS(Var)

% Random number generation according variables in the input Structure

% Generation of random values for
%   1) 'uni' Uniform distribution (Note: only natural numbers)
%   2) 'un2' Uniform distribution
%   3) 'nor' Normaly distributed random values with [Lower,Upper] limits
%   4) 'nol' Normaly distributed random values without limits
%   5) 'arr' Pick randomly elements from an array

% % -------------------------------------------------------------------------
% % ----- Inputs ----
% Var = Structure with Variability information, at least:
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
% % ----- Outputs ----
% randvalue = Generated random value
% % -------------------------------------------------------------------------

% % ---- Examples ----
% % Uniform distribtuion
%     randvalue = myrand('uni',1,3)
% % Normal distribution
%     randvalue = myrand('nor',1000,100,500,1500)

% % -------------------------------------------------------------------------

% **** Generation of random value ****

% -- For uniform distribution (Natural numbers) --
if strcmp(Var.dist,'uni')

    % Inputs
    mini = Var.var(1);
    maxi = Var.var(2);

    % Random value
    randvalue = (mini-1) + ceil((maxi-mini+1)*rand);

% -- For uniform distribution --
elseif strcmp(Var.dist,'un2')

    % Inputs
    mini = Var.var(1);
    maxi = Var.var(2);

    %Random value
    randvalue = mini + (maxi-mini)*rand;

% -- For normal distribution WITH limits --
elseif strcmp(Var.dist,'nor')

    % Inputs
    mini = Var.var(1);
    maxi = Var.var(2);
    mean_norm = Var.var(3);
    std_norm = Var.var(4);

    % Iterative process
    k_repeat = 1;
    while k_repeat == 1
    
        % Random value
        randvalue = mean_norm + std_norm * randn;
    
        if and(randvalue>=mini,randvalue<=maxi)
            k_repeat = 0;
        end % if and(randvalue>=mini,randvalue<=maxi)
    end % while k_repeat == 1

% -- For normal distribution NO limits --
elseif strcmp(Var.dist,'nol')

    % Inputs
    mean_norm = Var.var(1);
    std_norm = Var.var(2);

    % Random value
    randvalue = mean_norm + std_norm * randn;
        
% -- For random Array values --
elseif strcmp(Var.dist,'arr')
    
    % Inputs
    array_in = Var.var;
    
    n = length(array_in);

    if n == 1
        randvalue = array_in;
    else
        randvalue = randi(n);
        %randvalue = myrand('uni',1,n);
        randvalue = array_in(randvalue);
    end % if n == 1
    
end %if Var.dist

% ----- End of Script ----