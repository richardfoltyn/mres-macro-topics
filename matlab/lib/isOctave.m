function flag = isOctave()
%   ISOCTAVE() checks whether a program is being executed in GNU Octave.
%
%   FLAG = ISOCTAVE() returns true if a program in being executed in GNU
%       Octave, and false otherwise.
    flag = exist('OCTAVE_VERSION', 'builtin') ~= 0;
end
