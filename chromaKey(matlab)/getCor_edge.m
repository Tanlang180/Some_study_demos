function [Xcor,Ycor] = getCor_edge(a,b,edge_size)

for i = 1:edge_size
    Xcor(i) = a - floor(edge_size/2)+i-1;
    for j = 1:edge_size  
        Ycor(j) = b - floor(edge_size/2)+j-1;
    end
end
end