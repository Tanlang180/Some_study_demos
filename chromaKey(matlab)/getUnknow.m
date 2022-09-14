function temp = getUnknow(edge_mask,mask)
% get unknow region with more color spill pixels

height=size(edge_mask,1);
width=size(edge_mask,2);
se1=strel('square',5);
unknow = imdilate(edge_mask,se1);
temp = zeros(height,width);
lambda = 0.01;  % exit (0,1), 表示包含的色度溢出像素的阈值
for i = 1: height  
    for j = 1:width      
        if unknow(i,j)>0.01 && mask(i,j)>lambda
           temp(i,j) = 1;
        end
    end
end

end