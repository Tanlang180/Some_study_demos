function edge = getEdge(mask)

height = size(mask,1);
weight = size(mask,2);
edge = zeros(height,weight);
for i = 1:height
    for j = 1 : weight
        if mask(i,j)>0 && mask(i,j)<1
            edge(i,j) = 1;
        end
    end
end

end