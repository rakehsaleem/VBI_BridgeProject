function [Struct_out] = myMC_varDef(var_in,dist_in,text_in)

% Shorter version of the command:
%   struct('var',[50,120,100,10]/3.6,'dist','nor','text','Vehicle velocity (m/s)');

% % -------------------------------------------------------------------------
% % ----- Inputs ----
% var_in = array that defines the values for the chosen distribution
% dist_in = String indicating the name of the distribution
% text_in = String describing the current variability defined
% % ----- Outputs ----
% Struct_out = New structure with additional fields
%   .var = array that defines the values for the chosen distribution
%   .dist = String indicating the name of the distribution
%   .text = String describing the current variability defined
% % -------------------------------------------------------------------------

Struct_out.var = var_in;
Struct_out.dist = dist_in;
Struct_out.text = text_in;

% ---- End of script ----