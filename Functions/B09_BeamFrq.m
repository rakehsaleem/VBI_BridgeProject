function [Beam] = B09_BeamFrq(Calc,Beam)

% Calculates the beam frequencies given its system matrices

% Copyright (c) 2019, Daniel Cantero. All rights reserved.
% Use, copy or distribution of this program is only allowed with the 
% explicit authorization from the author. Please contact daniel.cantero@ntnu.no

% -------------------------------------------------------------------------
% ---- Input ----
% Beam = Structure with Beam's variables, including at least:
%   .SysM.K = Global Stiffness matrix
%   .SysM.M = Global Mass matrix
%   .BC.num_DOF_fixed = Number of fixed DOF
%   .Prop.Lb = Beam length
% Calc = Structre with calculation variables. It should include at least:
%   .Opt.beam_frq = Flag indication if frequencies should be calculated
%   .Opt.beam_modes = Flag indication if modes of vibration should be calculated
%   .Plot.P1_Beam_frq = If defined, plots distribution of natural frequencies
%   .Plot.P2_Beam_modes = If defined, plots modes of vibration of beam. The number of
%       modes to plots is defined by the value of P2_Beam_modes
% ---- Output ----
% Beam = Addition of fields to structure Beam:
%   .Modal.w = Circular frequencies of the beam
%   .Modal.f = Natural frequencies of the beam
%   .Modal.modes = Modes of vibration of in columns
% -------------------------------------------------------------------------

% Only natural frequencies calculation
if and(Calc.Opt.beam_frq == 1,Calc.Opt.beam_modes == 0)
    
    lambda = eig(full(Beam.SysM.K),full(Beam.SysM.M));
    Beam.Modal.w = sqrt(lambda);
    Beam.Modal.f = Beam.Modal.w/(2*pi);

    % Removing values associated to BC
    Beam.Modal.w = Beam.Modal.w(Beam.BC.num_DOF_fixed+1:end);
    Beam.Modal.f = Beam.Modal.f(Beam.BC.num_DOF_fixed+1:end);
    
% Natural frequencies and Modes of vibration calculation
elseif and(Calc.Opt.beam_frq == 1,Calc.Opt.beam_modes == 1)
    
    [V,lambda] = eig(full(Beam.SysM.K),full(Beam.SysM.M));       % Fastest
    %[V,lambda] = eig(inv(full(Beam.SysM.M))*full(Beam.SysM.K)); % Slower alternative
    %[V,lambda] = eig(full(Beam.SysM.K)/(full(Beam.SysM.M)));    % Slower alternative
    [lambda,k] = sort(diag(lambda));
    V = V(:,k); 

    % Normaliztion of eigenvectors
    Factor = diag(V'*Beam.SysM.M*V); 
    Beam.Modal.modes = V/(sqrt(diag(Factor)));
    
    % EigenValues to Natural frequencies
    Beam.Modal.w = sqrt(lambda);
    Beam.Modal.f = Beam.Modal.w/(2*pi);
    
    % Removing values associated to BC
    Beam.Modal.w = Beam.Modal.w(Beam.BC.num_DOF_fixed+1:end);
    Beam.Modal.f = Beam.Modal.f(Beam.BC.num_DOF_fixed+1:end);
    Beam.Modal.modes(:,1:Beam.BC.num_DOF_fixed) = [];
    
end % if and(Calc.beam_frq == 1,Calc.beam_frq == 0)

% ------------ Plotting ------------

% -- Plotting of calculated Natural frequencies --
if isfield(Calc.Plot,'P1_Beam_frq')
    figure; plot((1:length(Beam.Modal.f)),Beam.Modal.f,'.'); axis tight;
    %figure; semilogy((1:length(Beam.Modal.f)),Beam.Modal.f,'.'); axis tight;
        xlabel('Mode number'); ylabel('Frequency (Hz)');
        title(['Beam Only (1st frq: ',num2str(round(Beam.Modal.f(1),2)),' Hz;',...
        blanks(1),'Last frq: ',num2str(round(Beam.Modal.f(end),2)),' Hz)']);
    drawnow;
end % if isfield(Calc.Plot,'P1_Beam_frq')	

% -- Plotting Mode shapes --
if isfield(Calc.Plot,'P2_Beam_modes')
    if Calc.Plot.P2_Beam_modes > 0
        aux1 = ceil(Calc.Plot.P2_Beam_modes/2);
        figure; 
        for k = 1:Calc.Plot.P2_Beam_modes
            aux2 = max(abs(Beam.Modal.modes(1:2:end,k)));
            subplot(aux1,2,k)
            if exist('Beam_redux','var')
                Xdata = Beam_redux.acum;
            else
                Xdata = Beam.Mesh.Ele.acum;
            end % if exist('Beam_redux','var');
            plot(Xdata,Beam.Modal.modes(1:2:end,k)/aux2);
            hold on; plot(Xdata,-Beam.Modal.modes(1:2:end,k)/aux2,'r');
            plot(Beam.BC.loc,zeros(size(Beam.BC.loc)),'r.','MarkerSize',10);
            xlim([0,Beam.Prop.Lb]);
            title(['Mode ',num2str(k),' (',num2str(round(Beam.Modal.f(k),3)),' Hz)']);
        end % for k = 1:Calc.Plot.P2_Beam_modes
        drawnow;
    end % if Calc.Plot.P2_Beam_modes > 0
end % if isfield(Calc.Plot,'P2_Beam_modes')

% ---- End of function ----