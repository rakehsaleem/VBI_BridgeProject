% ---- Definition of all variables in VBI_DC_v2019 model ----

% In here, there is a description of all the variables used in the VBI_DC_v2019 model
% These variables are either inputs, outputs or both to different functions in the model.
% Note that all variables are grouped in structure variables.

% Copyright (c) 2019, Daniel Cantero. All rights reserved.
% Use, copy or distribution of this program is only allowed with the 
% explicit authorization from the author. Please contact daniel.cantero@ntnu.no

%--------------------------------------------------------------------------

% **** Calc ****
% Calc = Structure with calculation variables and options, including at least:
% Calc.Profile = Substructure with information about the road profile
%   Calc.Profile.type = Type of Profile 
%           -1 = Load an existing profile
%           0 = Smooth profile
%           2 = ISO random profile
%           3 = Step profile
%           4 = Ramp profile
%           5 = Sine Wave profile
% Depending on the profile type, additional variables should be defined
% for Calc.Profile.type = 2
% 	Calc.Profile.Info.class = ISO class in text format ('A','B',...)
%   Calc.Profile.Info.Gd_limtis = Array with the Gd limits for each class as indicated in ISO code
%   Calc.Profile.Info.Gd = Actual Gd value of the profile generated
%   Calc.Profile.Info.PSD_x = Frequency values of the PSD (x-coordinate)
%   Calc.Profile.Info.PSD_Y = Magnitude values of the PSD (y-coordinate)
%   Calc.Profile.Opt.class_var = flag to consider variation of roughness within the profile class (1 = yes)
% for Calc.Profile.type = 3
%   Calc.Profile.step_loc_x = Step location in x-coordinate system [m]
%   Calc.Profile.step_height = Step height in [m]
% for Calc.Profile.type = 4
%   Calc.Profile.ramp_loc_x_0 = Ramp start location in x-coordinate system [m]
%   Calc.Profile.ramp_loc_x_end = Ramp end location in x-coordinate system [m]    
%   Calc.Profile.ramp_height = Ramp height in [m]
% for Calc.Profile.type = 5
%   Calc.Profile.sine_wavelength = Wavelength of the sinusoidal [m]
%   Calc.Profile.sin_Amp = Amplitude of the sinusoidal [m]
%   Calc.Profile.sin_phase = Phase of sinusoidal at x=0 [rad]
% Calc.Profile.Load = Substructure with information about profile loading from a file
%   Calc.Profile.Load.on = Flag to switch on/off the profile loading (1=loading)
%   Calc.Profile.Load.file_name = String specifying the name of the file to load
% Calc.Profile.Spatial_frq = Substructure with information about the spatial frequency 
%       relevant when the profile is generated from ISO PSD (type 2)
% Calc.Profile.Save = Substructure with information about profile saving to a file
%   Calc.Profile.Save.on = Flag to switch on/off the profile saving (1=saving)
%   Calc.Profile.Save.file_name = String specifying the name of the file to save
%   Calc.Profile.Save.path = Path indicating the directory where to save the file
% Calc.Profile.dx = Spatial sampling distance on X direction [m]
% Calc.Profile.L = Total length of generated profile on X direcction [m]
% Calc.Profile.needed_x0 = Furthest to the left x coordinate needed for this event [m]
% Calc.Profile.needed_x_end = Furthest to the right x coordinate needed for this event [m]
% Calc.Profile.needed_L = Total length of profile needed for this event [m]
% Calc.Profile.x = Vector with x coordinates of the generated profile
% Calc.Profile.num_x = Number of sample points of the generated profile
% Calc.Profile.h = Vector with elevation (y coordinates) of the generated profile
% Calc.Solver = Substructure with information about the solution procedure
%   Calc.Solver.min_t_steps_per_second = Minimum time step per second
%   Calc.Solver.max_accurate_frq = Maximum accurate frequency of the solution
%   Calc.Solver.min_Beam_modes_considered = Number of beam modes to be considered for time step selection
%   Calc.Solver.t_end = Total time to be simulated
%   Calc.Solver.t_steps_per_second = Time steps per second
%   Calc.Solver.t_steps_criteria = Code number of the final adopted step criteria
%   Calc.Solver.t_steps_criteria_text = String with the text of the adopted step criteria
%       1 = Maximum vehicle frequency
%       2 = Defined maximum bridge modes considered
%       3 = Defined minimum steps per second
%       4 = Profile maximum frequency (spatial frequency x Vehicle velocity)
%       5 = User-defined maximum accurate frequency
%   Calc.Solver.t = Array of time steps to simulate
%   Calc.Solver.dt = Sampling period of t
%   Calc.Solver.num_t = Total number of time steps
%   Calc.Solver.t0_ind_beam = First time step for the first vehicle to enter the beam
%   Calc.Solver.t_end_ind_beam = Last time step for the first vehicle to enter the beam
%   Calc.Solver.num_t_beam = Number of time steps on the beam
%   Calc.Solver.t_beam = Array of time steps whiler vehicles on the beam
% Calc.Solver.NewMark = Substructure with information about the Newmark-Beta integration scheme
%   Calc.Solver.NewMark.damp = Selection of NewMark-Beta scheme
%       0 = No numerical damping
%       1 = With numerical damping
%   Calc.Solver.NewMark.beta = Constant of the Newmark-Beta scheme (0.25 for Average acceleration method = inconditionally stable)
%   Calc.Solver.NewMark.delta = Constant of the Newmark-Beta scheme (0.5 for Average acceleration method = inconditionally stable)
% Calc.Proc = Information about the solution procedure to solve the vehicle-bridge interaction
%   Calc.Proc.name = String with the name of the procedure
%   Calc.Proc.code = Code number of the procedure to use
%       1 = FI = Full iteration
%       2 = SSI = Step-by-step iteration
%       3 = Coup = Coupled solution
%   Calc.Proc.Iter.max_num = Maximum number of iterations (Default = 10)
%   Calc.Proc.Iter.criteria = Iteration criteria flag based on:
%       1 = deformation under wheels (default)
%   	2 = BM of whole beam
%   Calc.Proc.Iter.tol = Tolerance for iteration stopping criteria (Default = Calc.Cte.tol)
% Calc.Opt = Substructures with information about different options
%   Calc.Opt.verbose = Display detailed information about the current simulation
%   Calc.Opt.show_progress_every_s = Display calculation progress every X seconds
%   Calc.Opt.beam_frq = Calculate beam's frequencies (1=on)
%   Calc.Opt.beam_modes = Calculate beam's modes of vibration (1=on)
%   Calc.Opt.veh_frq = Calculate vehicle's frequencies (1=on)
%   Calc.Opt.vehInitSta = Calculate initial static deformation of vehicle (1=on)
%   Calc.Opt.beamInitSta = Calculate initial static deformation of beam  (1=on)
%   Calc.Opt.calc_mode_BM = How to calculate the BM on the beam
%       0 = Calculations done node by node
%       1 = Average results at the node 
%   Calc.Opt.calc_mode_Shear = How to calculate the Shear on the beam
%       0 = Calculations done node by node
%       1 = Average results at the node 
%   Calc.Opt.VBI = Flag idicating to switch on/off the vehicle-bridge interaction (1=on)
%   Calc.Opt.free_vib_seconds = Additional number of seconds in free vibration to include in the simulation
%   Calc.Opt.include_coriolis = Flag to include coriolis effects (1=on)
% Calc.Plot = Substructure with indicating what figures to plot at the end of the simulation
%   Calc.Plot.P1_Beam_frq = Plot the distribution of beam frequencies
%   Calc.Plot.P2_Beam_modes = Plot the first n modes of vibration (of Beam)
%       The number of modes to plots is defined by the value of P2_Beam_modes
%   Calc.Plot.P3_VehPos = Plot the vehicle position (velocity and acceleration)
%       A figure is shown with 3 subplots: 1) Load position in time; 2) Load velocity in time; 3) Load acceleration in time
%   Calc.Plot.P4_Profile = Plot the profiles and 1st derivative
%   Calc.Plot.Profile_original = Plot the original profile generated inside function B19
%   Calc.Plot.P5_Beam_U = Contourplot of Beam deformation
%   Calc.Plot.P6_Beam_U_under_Veh = Deformation under the vehicle
%   Calc.Plot.P7_Veh_U_iter = Vehicle total displacement for each iteration (Only FI procedure)
%   Calc.Plot.P8_Veh_Urel_iter = Vehicle relative displacement for each iteration (Only FI procedure)
%   Calc.Plot.P9_Beam_U_under_Veh_iter = Deformation under the vehicle for each iteration (Only FI procedure)
%   Calc.Plot.P10_diff_iter = The difference between solutions (Iteration criteria) (Only FI procedure)
%   Calc.Plot.P11_Beam_U_static = Calculates the Static Deformation of Beam (Due to Interaction force)
%   Calc.Plot.P13_Interaction_Force = Final interaction force
%   Calc.Plot.P14_Interaction_Force_iter = Interaction force for each iteration (Only FI procedure)
%   Calc.Plot.P16_Beam_BM = Contourplot of Beam BM
%   Calc.Plot.P17_Beam_BM_static = Contourplot of Beam Static BM
%   Calc.Plot.P18_Beam_Shear = Contourplot of Beam Shear
%   Calc.Plot.P19_Beam_Shear_static = Contourplot of Beam Static Shear
%   Calc.Plot.P20_Beam_MidSpan_BM = Static and Dynamic BM at mid-span
%   Calc.Plot.P21_Beam_MidSpan_BM_iter = Static and Dynamic BM at mid-span for various iterations (Only FI procedure)
%   Calc.Plot.P22_SSI_num_iterations = Number of iterations for each time step (Only for SSI procedure)
%   Calc.Plot.P23_vehicles_responses = Vehicles DOF responses (Displacement, velocity and acceleration)
%   Calc.Plot.P24_vehicles_wheel_responses = Responses at the wheels of the vehicle
%   Calc.Plot.P25_vehicles_under_responses = Responses under the wheel (Bridge response at the contact point)
%   Calc.Plot.P26_PSD_Veh_A = PSD of vehicle accelerations
%   Calc.Plot.P27_Veh_A = Time histories of vehicle accelerations
% Calc.Cte = Substructure with constants
%   Calc.Cte.grav = Gravity acceleration [m/s^2] (Use -9.81)
%   Calc.Cte.tol = Treshold under which two numerical values are considered identical

% -------------------------------------------------------------------------

% **** Veh ****
% Veh = Structure with the variables for all the vehicles
%               The information for i-th vehicle is stored in Veh(i)
% Veh.Model = Substructure with information about the vehicle model to use
%   Veh.Model.type = Name of function to run to generate the vehicle's system matrices
%   Veh.Model.function_path = Path of the location of the vehicle models definitions
% Veh.Prop = Substructure with information about the particular vehicle properties
%   Veh.Prop.mBi = Vector of body masses; [mB1, mB2, ...]
%   Veh.Prop.IyBi = Vector of body momments of intertia; [IyB1, IyB2, ...]
%   Veh.Prop.kTi = Vector of tyre stiffness; [kT1, kT2, ...]
%   Veh.Prop.cTi = Vector of tyre viscous damping; [cT1, cT2, ...]
%   Veh.Prop.kSi = Vector of suspension stiffness; [kS1, kS2, ...]
%   Veh.Prop.cSi = Vector of suspension viscous damping; [cS1, ...]
%   Veh.Prop.mSi = Vector of suspension mass; [mS1, mS2, ...]
%   Veh.Prop.mSi = Vector of suspension moment of inertia (if it is an axle group); [IS1, IS2, ...]
%   Veh.Prop.afBi = Vector of distance of body mass centre of gravity to the front of the body; [afB1, afB2, ...]
%   Veh.Prop.abBi = Vector of distance of body mass centre of gravity to the back of the body; [abB1, abB2, ...]
%   Veh.Prop.ei = Vector of x coordinate of each axle; [e1, e2, ...]
%   Veh.Prop.di = Vector of x coordinate of each tyre with respect to the centre of the axle group; [d1, d2, ...]
%   Veh.Prop.ax_sp = Vector of spacing of each tyre to the next; [0 ax_sp2, ax_sp3, ...]
%   Veh.Prop.ax_dist = Vector of distance of each tyre to the 1st tyre; [0 ax_dist2, ax_dist3, ...]
%   Veh.Prop.wheelbase = Distance between 1st and last tyre
%   Veh.Prop.num_wheels = Total number of wheels
% Veh.Pos = Substructure with information about the vehicle position, time and profile
%   Veh.Pos.x0 = Initial position of vehicle (from left bridge support)
%   Veh.Pos.vel = Velocity of vehicle [m/s]
%               Positive value = vehicle moving left to right
%               Negative value = vehicle moving right to left
%   Veh.Pos.prof_x0 = X coordinate of the tyre furthest away from the beam at start of event
%   Veh.Pos.prof_x_end = X coordinate of the tyre furthest away from the beam at end of event
%   Veh.Pos.min_t_end = Minimum time requried to simulate the crossing of the vehicle
%   Veh.Pos.wheels_x = Matrix for X coordinate for each wheel in time; size = [num_wheels,num_t]
%   Veh.Pos.wheels_on_beam = Matrix for flags indicating the presence on the beam of each wheel in time; size = [num_wheels,num_t]
%   Veh.Pos.t0_ind_beam = time step number when the firt wheel enters the beam
%   Veh.Pos.t_end_ind_beam = time step number when the last wheel enters the beam
%   Veh.Pos.wheels_h = Matrix for profile elevations for each wheel in time; size = [num_wheels,num_t]
%   Veh.Pos.wheels_hd = Matrix for profile elevations first derivative in time for each wheel in time; size = [num_wheels,num_t]
%   Veh.Pos.elexj = The beam element number for each wheel in time
%   Veh.Pos.xj = The relative position with respect to start of the beam element elexj for each wheel in time
% Veh.Event = Substructure with information about the event
%   Veh.Event.max_vel = Maximum vehicle velocity of all the vehicles in the event
%   Veh.Event.num_veh = Number of vehicles involved in the event
% Veh.DOF = Substructure with information about the DOF of the vehicle model
%               The information about the j-th DOF is stored in Veh.DOF(j)
%   Veh.DOF.dependency = Text specifiying if that DOF is dependent/independent
%   Veh.DOF.name = Text with the name of the DOF
%   Veh.DOF(1).num_dependent = number of dependent DOFs (Note, only for DOF(1))
%   Veh.DOF(1).num_independent = number of independent DOFs (Note, only for DOF(1))
%   Veh.DOF.type = Text stating the type of DOF (displacement/rotational)
% Veh.SysM = Substructure with the vehicle's system matrices
%   Veh.SysM.M = Mass matrix of the vehicle model
%   Veh.SysM.C = Damping matrix of the vehicle model
%   Veh.SysM.K = Stiffness matrix of the vehicle model
%   Veh.SysM.N2w = Matrix that relates the nodal displacements (DOF) to the displacements of the 
%               top of the wheels. This is relevant for vehicle with axle groups
% Veh.Static = Substructure with information about the vehicle's static configuration
%   Veh.Static.F_vector_no_grav = Vector of masses to use to calculate the static
%               load of the vehicle; size = [1, num_DOF]. Multiply it by gravity to get the force
%               to apply to each DOF
%   Veh.Static.load = Vector of static vertical forces on the road; size = [num_wheels, 1]
%   Veh.Static.check = Internal check result of static force calculation (1=OK)
% Veh.Modal = Substructure containing the results from the modal analysis of the vehicle
%   Veh.Modal.w = Vector of circular frequencies of vehicle; size = [1, num_DOF]
%   Veh.Modal.f = Vector of frequencies of vehicle; size = [1, num_DOF]

% -------------------------------------------------------------------------

% **** Beam ****
% Beam = Structure with information about the beam and its modelling
% Beam.Prop = Substructure with information about the beam's mechanical properties
%   Beam.Prop.type = Specify the type of beam bridge
%           0 = Custom definition of properties
%           'T' = T type beam bridge
%           'Y' = Y type beam bridge
%           'SY' = Super-Y type beam bridge
%   Beam.Prop.Lb = Span of beam [m]
%   Beam.Prop.rho = Mass per unit length [kg/m]
%   Beam.Prop.E = Modulus of elasticity [N*m^2]
%   Beam.Prop.I = Second moment of area [m^4]
%   Beam.Prop.A = Cross sectional area [m^2]
%   Beam.Prop.damp_per = Percentage of bridge damping
%   Beam.Prop.damp_xi = Damping ratio (damp_per/100)
%   Beam.Prop.Prop.E_n = Beam Young's Modulus
%       Vector of values giving the E for each element; size = [1,num. of elements]
%   Beam.Prop.Prop.I_n = Beam's section Second moment of Inertia product
%       Vector of values giving the I for each element; size = [1,num. of elements]
%   Beam.Prop.Prop.rho_n = Beam's density
%       Vector of values giving the rho for each element; size = [1,num. of elements]
%   Beam.Prop.Prop.A_n = Beam's section Area
%       Vector of values giving the A for each element; size = [1,num. of elements]
% Beam.BC = Substructure with information about the beam's boundary conditions
%   Beam.BC.loc = Vector for locations of supports in X direction; [xBC1, xBC2, ...]
%   Beam.BC.vert_stiff = Vertical stiffnes value of each of the supports
%           -1 =    Fixed, no displacement
%           0 =     Free vertical displacement
%           value = Vertical stiffness of the support
%   Beam.BC.rot_stiff = Rotational stiffnes value of each of the supports
%           -1 =    Fixed, no rotation
%           0 =     Free rotation
%           value = Rotational stiffness of the support
%   Beam.BC.supp_num = Number of supports
%   Beam.BC.loc_ind = Node number closest to the support
%   Beam.BC.DOF_fixed = Vector of DOFs with fixed boundary condition
%   Beam.BC.DOF_with_values = Vector of DOFs that have some additional stiffness
%   Beam.BC.DOF_stiff_values = Vector of stiffness values to be added to the DOF specified in Beam.BC.DOF_with_values
%   Beam.BC.num_DOF_fixed = Number of fixed DOFs
%   Beam.BC.num_DOF_with_values = Number of DOFs with additional stiffness
%   Beam.BC.DOF_fixed_value = Value to use when DOF is fixed. (See comment in B02)
% Beam.Mesh = Substructure with information about the beam mesh
% Beam.Mesh.Ele = Substructure with information about the elements of the mesh
%   Beam.Mesh.Ele.num = Total number of elements in the model
%   Beam.Mesh.Ele.a = Vector with each element X dimension
%   Beam.Mesh.Ele.acum = X coordinate of each node
%   Beam.Mesh.Ele.nodes = Matrix of size [number of elements, num. nodes per element]
%           Each row includes the indices of the nodes for each element. 
%           These variable is more useful when using more complex elements (not included here)
%   Beam.Mesh.Ele.DOF = Matrix of size [number of elements, num. DOF per element]
%           Each row includes the DOF asociated to every element. Each element represents a row.
%   Beam.Mesh.Ele.shape_fun = Anonymous function of the element's shape function.
%           Inputs [x = local x coordinate, a =element size]
%   Beam.Mesh.Ele.shape_fun_p = Anonymous function of the first derivative (prime) element's shape function. 
%           Inputs [x = local x coordinate, a =element size]
%   Beam.Mesh.Ele.shape_fun_pp = Anonymous function of the second derivative (double-prime) element's shape function. 
%           Inputs [x = local x coordinate, a =element size]
% Beam.Mesh.Node = Substructure with information about the nodes of the mesh
%   Beam.Mesh.Node.at_mid_span = Node at mid-span
%   Beam.Mesh.Node.coord = Coordinates of all nodes [X coord], one row for each node.
%   Beam.Node.num_perEle = Nodes Per Element
%   Beam.Mesh.Node.num = Total number of nodes
% Beam.Mesh.DOF = Substructure with information about the DOFs of the mesh
%   Beam.Mesh.DOF.num = Total number of DOF of the mesh
%   Beam.Mesh.DOF.num_perNode = Number of DOF per node
% Beam.Damage = Substructure with information about the beam damage
%   Beam.Damage.type = Number indicating the type of damage
%           0 = No damage
%           1 = One element damage
%           2 = Global damage
%   Beam.Damage.E.per = Vector for local stiffnes reduction = [Location in m, Magnitude of damage in %] 
%           For global damage input = [ -unused- , Magnitude of damage in %]
%   Beam.Damage.I.per = Vector for local Inertia reduction = [Location in m, Magnitude of damage in %]
%           For global damage input = [ -unused- , Magnitude of damage in %]
%   Beam.Damage.rho.per = Vector for local density reduction = [Location in m, Magnitude of damage in %]
%           For global damage input = [ -unused- , Magnitude of damage in %]
%   Beam.Damage.A.per = Vector for local area reduction = [Location in m, Magnitude of damage in %]
%           For global damage input = [ -unused- , Magnitude of damage in %]
% Beam.SysM = Substructure with information about beam's system matrices
%   Beam.SysM.K = Global stiffness matrix
%   Beam.SysM.C = Global damping matrix
%   Beam.SysM.M = Global mass matrix
% Beam.SysM.Opt = Substructure with information about beam's system matrices options
%   Beam.SysM.Opt.Mconsist_value = Option to select the type of Mass matrix
%           1 = Consitent mass matrix (Default)
%           2 = Lumped mass matrix
% Beam.Modal = Substructure with information about beam's modal analysis
%   Beam.Modal.num_rigid_modes = Number of rigid modes
%   Beam.Modal.modes = Modes of vibration of in columns; size = [Num. of DOF, Num. of modes]
%   Beam.Modal.w = Circular frequencies of the beam (column vector)
%   Beam.Modal.f = Natural frequencies of the beam (column vector)

% -------------------------------------------------------------------------

% **** Sol ****
% Sol = Structure containing the results from the simulation
% Sol.Veh = Substructure containing results for the vehicle(s)
%           The information for i-th vehicle is stored in Sol.Veh(i)
%   Sol.Veh.U = Vehicle's DOF displacements (or rotations)
%           size = [Num of vehicle DOFs, Num. of simulation time steps]
%   Sol.Veh.V = Vehicle's DOF velocities
%           size = [Num of vehicle DOFs, Num. of simulation time steps]
%   Sol.Veh.U = Vehicle's DOF accelerations
%           size = [Num of vehicle DOFs, Num. of simulation time steps]
% Sol.Veh.Under = Substructure containing results for the vehicle(s) wheel contact point
%           At the point where the tyre meets the road
%   Sol.Veh.Under.def = Beam deformation under the wheels
%           size = [Number of wheels, Num. of simulation time steps on the beam]
%   Sol.Veh.Under.vel = Beam's velocity under the wheels
%           size = [Number of wheels, Num. of simulation time steps on the beam]
%   Sol.Veh.Under.h = Profile elevation under the wheels
%           size = [Number of wheels, Num. of simulation time steps on the beam]
%   Sol.Veh.Under.hd = 1st time derivative of profile elevation under the wheels
%           size = [Number of wheels, Num. of simulation time steps on the beam]
%   Sol.Veh.Under.onBeamF = Force of each wheel acting on the beam
%           size = [Number of wheels, Num. of simulation time steps on the beam]
% Sol.Veh.Wheels = Substructure containing results for the vehicle(s) wheel displacement and velocities
%           These are the vehicle responses at the top of the tyre
%           For wheels not in an axle group this is the same as the suspension displacement and velocities
%   Sol.Veh.Wheels.U = Top of the tyre displacements
%           size = [Num of wheels, Num. of simulation time steps]
%   Sol.Veh.Wheels.Urel = Relative displacements between top and bottom of tyre (Wheels' relative displacements)
%           size = [Num of wheels, Num. of simulation time steps]
%   Sol.Veh.Wheels.V = Top of the tyre velocities
%           size = [Num of wheels, Num. of simulation time steps]
%   Sol.Veh.Wheels.Vrel = Relative velocities between top and bottom of tyre (Wheels' relative velocities)
%           size = [Num of wheels, Num. of simulation time steps]
% Sol.Beam = Substructure with results for the beam
% Sol.Beam.(LoadEffect) = Substructure with "LoadEffect" results for the beam
%           Several load effects are considered, with following field names:
%           U = Deformation
%           V = Velocity
%           A = Acceleration
%           BM = Bending moment
%           Shear = Shear forces
%           U_static = Static deformation
%           BM_static = Static bending moment
%           Shear_static = Static shear forces
%   Sol.Beam.(LoadEffect).value_DOFt = Result for each DOF in time; size = [num of DOF, num. of simulation time step on beam]
%   Sol.Beam.(LoadEffect).value_xt = Vertical displacement for each node in time; size = [num of DOF, num. of simulation time step on beam]
%   Sol.Beam.(LoadEffect).Max.value = Maximum value
%   Sol.Beam.(LoadEffect).Max.COP = Location of maximum along the beam (Critical observation point)
%   Sol.Beam.(LoadEffect).Max.pCOP = COP in percentage of beam length
%   Sol.Beam.(LoadEffect).Max.cri_t_in = Time step when the maximum occurred
%   Sol.Beam.(LoadEffect).Max.value05 = Maximum value at mid-span
%   Sol.Beam.(LoadEffect).Min.value = Minimum value
%   Sol.Beam.(LoadEffect).Min.COP = Location of minimum along the beam (Critical observation point)
%   Sol.Beam.(LoadEffect).Min.pCOP = COP in percentage of beam length
%   Sol.Beam.(LoadEffect).Min.cri_t_in = Time step when the minimum occurred
%   Sol.Beam.(LoadEffect).Min.value05 = Minimum value at mid-span

% -------------------------------------------------------------------------
% **** Local variables in some functions ****

% **** Function B14 ****
% Fextnew = Equivalent nodal forces vector

% **** Function B20 ****
% ---- Input ----
% prof_x = X coordenates of the profile (m)
% prof_y = Matrix containing Y coordenates of the profiles in rows (m)
% wind = Window. Distance to be averaged, should be an even number, and in (m)
%       Recomendation: Use for wind dx multiplied by an EVEN number
% % ---- Outputs ----
% prof_y = New smoother profile

% **** Function B21 ****
% % ---- Input ----
% L = Length of profile on X direcction (m)
% dx = Spatial sampling distance on X direction
% Gd = Roughness coeficient in ISO form [m^3]
%       If Gd = value, this is the value to be used
%       If Gd = A, random Gd value for class A
%       If Gd = B, random Gd value for class B
%       If Gd = C, random Gd value for class C
%       If Gd = D, random Gd value for class D
%       If Gd = E, random Gd value for class E
%       If Gd = F, random Gd value for class F
%       If Gd = G, random Gd value for class G
%       If Gd = H, random Gd value for class H
% class_var = 
%       0 = Centre value for specified Gd range selected
%       1 = Random value of Gd within the profile class selected
% spaf_min = Minimum spatial frequency to be considered
% spaf_max = Maximum spatial frequency to be considered
% spaf_ref = Reference spatial frequency for the definition of ISO PSD
% Gd_limits = As defined in code
% ---- Output ----
% prof_x = X values of the profile
% prof_y = Y values of the profile
% Gd = Roughness coeficient used
% ISO_spaf = X coordinates of ISO PSD (Spatial frequency)
% ISO_PSD = Y coordinates of ISO PSD [m^3]

% **** Function B28 ****
% -- Optional inputs --
% varargin{1} = 0 = Only vehicle passing on the approach is calculated (default)
%             = 1 = then calculation also includes the vehicle crossing
%                   beam (and free vibrations). It calculates all time steps

% **** Function B30 ****
% % ---- Input ----
% L = Length of Beam element
% E = Young's Modulus of the beam element
% I = Moment of inertia of the beam element
% % ---- Output ----
% He = Element Stress-Displacement matrix

% **** Function B32 ****
% % ---- Input ----
% L = Length of Beam element
% E = Young's Modulus of the beam element
% I = Moment of inertia of the beam element
% % ---- Output ----
% HSe = Element Stress-Displacement matrix

% **** Function B31 ****
% ---- Optional inputs ----
% static_flag = Calculate the static BM if this optional input is equal to 1

% **** Function B33 ****
% ---- Optional inputs ----
% static_flag = Calculate the static shear if this optional input is equal to 1

% ---- End of script ----