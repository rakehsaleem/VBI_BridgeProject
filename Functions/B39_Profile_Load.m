function [Calc] = B39_Profile_Load(Calc)
% If selected, loads a previously saved road profile
% (Cross-platform safe: normalizes paths and checks file/variable)

% ---- Normalize path & filename ----
p = Calc.Profile.Load.path;
f = Calc.Profile.Load.file_name;

% Convert to char for safety
if isstring(p), p = char(p); end
if isstring(f), f = char(f); end

% Replace backslashes with the OS separator; collapse doubles; trim trailing
p = strrep(p, '\', filesep);
p = regexprep(p, '[/\\]+', filesep);
if ~isempty(p) && p(end) == filesep
    p(end) = [];
end

% Ensure .mat extension on filename if missing
[~,~,ext] = fileparts(f);
if isempty(ext)
    f = [f '.mat'];
end

% Build full path
fullPath = fullfile(p, f);

% ---- Existence checks ----
if ~exist(fullPath, 'file')
    error('B39_Profile_Load:FileNotFound', ...
        'Profile file not found:\n  %s\n(From path="%s", file="%s")', fullPath, p, f);
end

% ---- Load only the needed variable ----
S = load(fullPath, 'Profile');  % loads into struct S.Profile
if ~isfield(S, 'Profile')
    % If file exists but variable isn't named Profile, fall back to raw load
    T = load(fullPath);
    vars = fieldnames(T);
    if isempty(vars)
        error('B39_Profile_Load:EmptyFile', ...
            'File loaded but contains no variables: %s', fullPath);
    end
    % Try to guess: use the first variable as Profile
    warning('B39_Profile_Load:ProfileMissing', ...
        ['Variable "Profile" not found in %s. Using "%s" instead. ' ...
         'Consider saving the file with a variable named Profile.'], ...
         fullPath, vars{1});
    Profile = T.(vars{1});
else
    Profile = S.Profile;
end

% ---- Validate expected fields ----
reqFields = {'L','dx','x','num_x','h','Spatial_frq'};
missing = reqFields(~isfield(Profile, reqFields));
if ~isempty(missing)
    error('B39_Profile_Load:BadProfile', ...
        'Loaded Profile is missing required fields: %s\nFile: %s', ...
        strjoin(missing, ', '), fullPath);
end

% ---- Original logic: check length and copy into Calc ----
if Calc.Profile.needed_L < Profile.L
    Calc.Profile.dx          = Profile.dx;
    Calc.Profile.L           = Profile.L;
    Calc.Profile.x           = Profile.x;
    Calc.Profile.num_x       = Profile.num_x;
    Calc.Profile.h           = Profile.h;
    Calc.Profile.Spatial_frq = Profile.Spatial_frq;

    if isfield(Profile,'Info')
        Calc.Profile.Info = Profile.Info;
    end
    if isfield(Profile,'Opt')
        Calc.Profile.Opt = Profile.Opt;
    end
else
    error('B39_Profile_Load:TooShort', ...
        ['Loaded profile length (L=%.6g) is too short for this event ' ...
         '(needed_L=%.6g). File: %s'], ...
         Profile.L, Calc.Profile.needed_L, fullPath);
end

end
