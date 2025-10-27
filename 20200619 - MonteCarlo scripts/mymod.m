function [b] = mymod(a,m)

% Is the same as mod() command but when a==m it outputs m (instead of zero)

% % -----------------------------------------------------------------------
% % ----- Inputs ----
% a = numerator
% m = denominator
% % ----- Output ----
% b = the remainder of the division a/m
%   but if a == m, the b = m
% % -----------------------------------------------------------------------

b = mod(a,m);

% if b == 0
%     b = m;
% end % if b == 0
% -- Alternative: Allows for b to be an array
b(b==0) = m;
    
% ---- End of function ----