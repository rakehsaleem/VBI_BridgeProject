function [Fextnew] = B14_EqVertNodalForce(Calc,Veh,Beam,Sol)

% Distributee the forces to the appropriate DOFs
% For each time step and depending on the location of the force, the actual
% forces must be distributed to the degrees of freedom, this can be
% accomplished through the shape functions.
% NOTE: Only Vertical forces calculated

% Copyright (c) 2019, Daniel Cantero. All rights reserved.
% Use, copy or distribution of this program is only allowed with the 
% explicit authorization from the author. Please contact daniel.cantero@ntnu.no

% -------------------------------------------------------------------------
% % ---- Inputs ----
% Beam = Structure with Beam's variables, including at least:
%   .Mesh.DOF.num = Total number of DOF
%   .acum = X coordinate of each node
%   .Mesh.Ele.nodes = Based on ele_nodes. Each row includes the DOF asociated to every element.
%       Each element represents a row.
%   .bc = Vector containing DOF with boundary conditions
% Calc = Structure with Calculation variables, including at least:
%   .tn = Total number of time steps
%   .F = External force for every time step
%   .x = Location of external for for every time step
% % ---- Outputs ----
% Fextnew = Equivalent nodal forces vector
% -------------------------------------------------------------------------

% ----------------------- Original implementation -------------------------

% Calculation options (Only for Point loads)
k_fun = 0;      % Calculate NOT using subfunction
%k_fun = 1;      % Calculate using subfunction

% Initialize variable
%Fextnew = sparse(Beam.Mesh.DOF.num,Calc.Solver.num_t_beam); % Apparently it is slower
Fextnew = zeros(Beam.Mesh.DOF.num,Calc.Solver.num_t_beam);

% Vehicles loop
for veh_num = 1:Veh(1).Event.num_veh

    % Wheel loop
    for wheel_num = 1:Veh(veh_num).Prop.num_wheels

        Fextnew1 = Fextnew*0;

        % Time loop
        for t = 1:Calc.Solver.num_t_beam

            elex = Veh(veh_num).Pos.elexj(wheel_num,Calc.Solver.t0_ind_beam-1+t);
            x = Veh(veh_num).Pos.xj(wheel_num,Calc.Solver.t0_ind_beam-1+t);
            if elex > 0

                % DOFs for the element    
                ele_DOF = Beam.Mesh.Ele.DOF(elex,:);

                % Multiplication of nodal displacements by corresponding shape function value
                if k_fun == 1
                    % --- Tidy alternative --- (A bit slower)
                    Fextnew1(ele_DOF,t) = Sol.Veh(veh_num).Under.onBeamF(wheel_num,t)*shapefunBeam(Beam.Mesh.Ele.a(elex),x);
                elseif k_fun == 0
                    % --- Alternative without function --- (A bit faster)
                    ai = Beam.Mesh.Ele.a(elex); 
                    Fextnew1(ele_DOF,t) = Sol.Veh(veh_num).Under.onBeamF(wheel_num,t)* ...
                        [ (ai+2*x)*(ai-x)^2/ai^3;       x*(ai-x)^2/ai^2;   x^2*(3*ai-2*x)/ai^3;      -x^2*(ai-x)/ai^2];
                end % if k_fun == 1        
            end % if elex > 0
        end % for t = 1:Calc.Solver.num_t_beam

        Fextnew = Fextnew + Fextnew1;

    end % for wheel_num = 1:Veh(veh_num).Prop.num_wheels
end % for veh_num = 1:Veh(1).Event.num_veh

% Application of boundary conditoins to force vector
Fextnew(Beam.BC.DOF_fixed,:) = 0;  % Vertical force = 0

% Function for Tidy alternative
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
function vect = shapefunBeam(a,x)
    % a = Element size in X direction
	% x = Position of force in X direction (local coordinates)
    
    vect = [ (a+2*x)*(a-x)^2/a^3;       x*(a-x)^2/a^2;   x^2*(3*a-2*x)/a^3;      -x^2*(a-x)/a^2];
   
%end         % function shapefunQ4
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

% % -------------------------- Sparse Alternative --------------------------- (Slightly slower)
% 
% % Initialize variables
% rows  = [];
% cols  = [];
% F_vals = [];
% ones_1_4 = ones(1,4);
% 
% % Vehicle loop
% for veh_num = 1:Veh(1).Event.num_veh
% 
%     % Wheel loop
%     for wheel = 1:Veh(veh_num).Prop.num_wheels
% 
%         % ---- Time loop ----
%         for t = 1:Calc.Solver.num_t_beam
% 
%             elex = Veh(veh_num).Pos.elexj(wheel,t); 
%             
%             if elex > 0
%                 
%                 x = Veh(veh_num).Pos.xj(wheel,t);
%                 a = Beam.Mesh.Ele.a(elex);
% 
%                 % DOFs for the element    
%                 ele_DOF = Beam.Mesh.Ele.DOF(elex,:);
% 
%                 rows = [rows, ele_DOF];
%                 cols = [cols, t*ones_1_4];
%                 %F_vals = [F_vals; Sol.Veh(veh_num).Under.onBeamF(wheel,t)*Beam.Mesh.Ele.shape_fun(x,a)];
%                 ai = Beam.Mesh.Ele.a(elex);
%                 F_vals = [F_vals; Sol.Veh(veh_num).Under.onBeamF(wheel,t)*...
%                         [ (ai+2*x)*(ai-x)^2/ai^3;       x*(ai-x)^2/ai^2;   x^2*(3*ai-2*x)/ai^3;      -x^2*(ai-x)/ai^2]];
%             end % if elex > 0
%         end % for t = 1:Calc.Solver.num_t
%     end % for wheel = 1:length(Ry)
% end % for veh_num = 1:Veh(1).Tnum
% 
% Fextnew = sparse(rows,cols,F_vals,Beam.Mesh.DOF.num,Calc.Solver.num_t_beam);
% 
% % Application of boundary conditoins to force vector
% Fextnew(Beam.BC.DOF_fixed,:) = 0;  % Vertical force = 0

% ---- End of function ----