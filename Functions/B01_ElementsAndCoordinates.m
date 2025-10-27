function [Beam] = B01_ElementsAndCoordinates(Beam)

% Calculation of Node coordinates and associated Nodes (and DOF) to each element.
% Specifies the property values (E,I,rho and A) for each element
% Also generates additional and auxiliary variables needed in the FEM model
 
% Copyright (c) 2019, Daniel Cantero. All rights reserved.
% Use, copy or distribution of this program is only allowed with the
% explicit authorization from the author. Please contact daniel.cantero@ntnu.no

% -------------------------------------------------------------------------
% % ---- Inputs ----
% Beam = Structure with Beam's variables, including at least:
%   .Prop.Lb = Lenght of beam
%   .Mesh.Ele.num = Number of elements in the mesh
% % ---- Outputs ----
% Beam = Addition of fields to structure Beam:
%   .Mesh.Ele.a = Vector with each element X dimension
%   .Mesh.Ele.acum = X coordinate of each node
%   .Mesh.Node.coord = Coordinates of all nodes [X coord], one row for each node
%   .Node.num_perEle = Nodes Per Element
%   .DOF.num_perNode = DOFs per node
%   .Mesh.Ele.num = Total number of elements in the model
%   .Mesh.Ele.nodes = Each row includes the indices of the nodes for each element. 
%       These variable is more useful in more complex elements like (not included here)
%   .Mesh.Ele.DOF = Each row includes the DOF asociated to every element. Each element represents a row.
%   .Mesh.Node.num = Total number of nodes
%   .Mesh.DOF.num = Total number of DOF
%   .Prop.E_n = Beam Young's Modulus
%       Array of values giving the E for each element (same size as Beam.Mesh.Ele.a)
%   .Prop.I_n = Beam's section Second moment of Inertia product
%       Array of values giving the I for each element (same size as Beam.Mesh.Ele.a)
%   .Prop.rho_n = Beam's density
%       Array of values giving the rho for each element (same size as Beam.Mesh.Ele.a)
%   .Prop.A_n = Beam's section Area
%       Array of values giving the A for each element (same size as Beam.Mesh.Ele.a)
% -------------------------------------------------------------------------

% Mesh definition 
Beam.Mesh.Ele.a = ones(1,Beam.Mesh.Ele.num)*Beam.Prop.Lb/Beam.Mesh.Ele.num;
Beam.Mesh.Ele.acum = [0 cumsum(Beam.Mesh.Ele.a)];

% -- Various definitions --
Beam.Mesh.Node.coord = Beam.Mesh.Ele.acum';
Beam.Mesh.Node.num_perEle = 2;
Beam.Mesh.DOF.num_perNode = 2;
Beam.Mesh.Ele.nodes = ones(length(Beam.Mesh.Ele.a),1)*(1:2);
Beam.Mesh.Ele.DOF = (1:2:(Beam.Mesh.Ele.num)*Beam.Mesh.Node.num_perEle)';
Beam.Mesh.Ele.DOF = [Beam.Mesh.Ele.DOF,Beam.Mesh.Ele.DOF+1,Beam.Mesh.Ele.DOF+2,Beam.Mesh.Ele.DOF+3];
Beam.Mesh.Node.num = size(Beam.Mesh.Node.coord,1);
Beam.Mesh.DOF.num = Beam.Mesh.Node.num*Beam.Mesh.Node.num_perEle;

% -- Element by element property definition Input processing --
% Beam Young's Modulus
if length(Beam.Prop.E) == 1
    Beam.Prop.E_n = ones(Beam.Mesh.Ele.num,1)*Beam.Prop.E;
end % if length(Beam.Prop.E) == 1
% Beam's section Second moment of Inertia product
if length(Beam.Prop.I) == 1
    Beam.Prop.I_n = ones(Beam.Mesh.Ele.num,1)*Beam.Prop.I;
end % if length(Beam.Prop.I) == 1
% Beam's density
if length(Beam.Prop.rho) == 1
    Beam.Prop.rho_n = ones(Beam.Mesh.Ele.num,1)*Beam.Prop.rho;
end % if length(Beam.Prop.rho) == 1
% Beam's section Area
if length(Beam.Prop.A) == 1
    Beam.Prop.A_n = ones(Beam.Mesh.Ele.num,1)*Beam.Prop.A;
end % if length(Beam.Prop.A) == 1
if length(Beam.Prop.h) == 1
    Beam.Prop.h_n = ones(Beam.Mesh.Ele.num,1)*Beam.Prop.h;
end % if length(Beam.Prop.A) == 1

% ---- End of function ----