function img = blengding(fg,alpha,bgColor,bg2)

height = size(fg,1); width = size(fg,2);
img = zeros(height,width,3);
bg1 = zeros(height,width,3);
bg1(:,:,1) = bgColor(1) * ones(height,width,1);
bg1(:,:,2) = bgColor(2) * ones(height,width,1);
bg1(:,:,3) = bgColor(3) * ones(height,width,1);


img(:,:,1) = blending2(fg(:,:,1), alpha,  bg1(:,:,1),bg2(:,:,1));
img(:,:,2) = blending2(fg(:,:,2), alpha,  bg1(:,:,2),bg2(:,:,2));
img(:,:,3) = blending2(fg(:,:,3), alpha,  bg1(:,:,3),bg2(:,:,3));

end

function matrix = blending2(fg,a,b1,b2)

matrix =double(fg + (1-a).*(b2-b1));

end