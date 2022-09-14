function img = greenProcess(img,edge_mask,mask,choice)
% choice==0 nomal green channel delay
% choice==1 select green channel impairment

red = img(:,:,1);
green = img(:,:,2);
blue = img(:,:,3);

switch(choice)
    case 1
        for i = 1:3
            edge_mask3d(:,:,i)=edge_mask;
        end
        imgEdge = img.*edge_mask3d;
             
        r = imgEdge(:,:,1);
        g = imgEdge(:,:,2);
        b = imgEdge(:,:,3);

        gi = img(:,:,2).*(1-edge_mask);
        go = g-0.5*(2*g-r-b);
        ga = gi+go;

        img(:,:,2) = ga;
    case 2
        height = size(img,1); width = size(img,2);d=size(img,3);
        for i = 1: height
            for j = 1:width
                if edge_mask(i,j)>0 && mask(i,j)>0.2
                    if  green(i,j)>blue(i,j) && green(i,j)>red(i,j)
                        img(i,j,2)=img(i,j,2)-0.5*abs(2*img(i,j,2)-img(i,j,1)-img(i,j,3));
                    end
                end
            end
        end
end
end

