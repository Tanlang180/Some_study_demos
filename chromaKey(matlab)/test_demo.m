%% HSV
hsv = rgb2hsv(fgB,fgG,fgB);
h = hsv(:,:,1);
s = hsv(:,:,2);
v = hsv(:,:,3);
imshow(hsv);
imshow(h);
imshow(s);
imshow(v./255);
imshow(fg./255);

%% matrix
background = 'images/background/bg2.jpg';
img = double(imread(background));
height = size(img,1);
width = size(img,2);

x = (img * 0.5) .* (img * 0.2);
y = zeros(height,width,3);
for i = 1 : height
    for j = 1 : width
        y(i,j,:) = (img(i,j,:)*0.5) .* (img(i,j,:)*0.2);
    end
end

out = x-y;
if out == zeros(height,width,3)
    disp('yes');
end



%% RGBA
rgba = imread('./images/foreground/default_fg.png');
imshow(rgba);