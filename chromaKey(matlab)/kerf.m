function feature = kerf(vector)
% author: tanlang
% illustration: compute the proximity 
%  Introducing lambda and alpha, Their values belong to [0,1], 
%  and the larger the values indicate the greater proportion of the feature when calculating proximity

r = vector(1,:);
g = vector(2,:);
b = vector(3,:);
x = vector(4,:);
y = vector(5,:);
lambda = 1;
alpha = 0.2;
feature = lambda*(r+g+b)./255 + alpha*(x+y);

end