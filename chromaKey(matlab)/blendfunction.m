%% Input data
% a - foreground,
% b - background,
% m - mask.
%% Output data
% z - blended.
function [ z ] = blendfunction(a, b, m)
z = double(a.*m + b.*(1 - m));
end
