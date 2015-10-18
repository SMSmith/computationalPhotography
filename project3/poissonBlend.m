function im_out = poissonBlend(im_s, mask_s, im_background)
    % Converting vector index to image index
    [h, w, ~] = size(im_background); 
    im2var = zeros(h, w); 
    im2var(1:h*w) = 1:h*w;
    
    % For finding gradients
%     laplacianFilter = [0 -1 0;
%                       -1 4 -1;
%                        0 -1 0];
    laplacianFilter = fspecial('laplacian',0);
    
    % Initialize to giant sparse set of zeros 
    Ax = ones(5*h*w,1); Ay = ones(5*h*w,1);
    Avalue = ones(5*h*w,1);
    b = zeros(h*w,1);
    
    im_out = zeros(h,w,3);
    
    % Look at each channel of the background and foreground images
%     sGradient = conv2(rgb2gray(im_s),laplacianFilter,'same');
    for channel = 1:3
        sGradient = conv2(im_s(:,:,channel), laplacianFilter, 'same');
        bGradient = conv2(im_background(:,:,channel), laplacianFilter, 'same');
        d = 1;
        for x=1:w
            for y=1:h
                e = im2var(y,x);
                if mask_s(y,x) == 1
                    % (Ax,Ay) are the locations of Avalue
                    % This is essentially a laplacian filter in the background image
                    Ay(d) = e; Ax(d) = im2var(y,x+1); Avalue(d) = 1;
                    Ay(d+1) = e; Ax(d+1) = im2var(y,x-1); Avalue(d+1) = 1; 
                    Ay(d+2) = e; Ax(d+2) = im2var(y+1,x); Avalue(d+2) = 1;
                    Ay(d+3) = e; Ax(d+3) = im2var(y-1,x); Avalue(d+3) = 1;
                    Ay(d+4) = e; Ax(d+4) = e; Avalue(d+4) = -4;
                    d=d+5;
                    % Mixed Gradients
                    mg = max(abs(sGradient(y,x)),abs(bGradient(y,x)));
                    if mg == abs(sGradient(y,x))
                        b(e) = sGradient(y,x);
                    else
                        b(e) = bGradient(y,x);
                    end
                    % Source gradient
%                     b(e) = sGradient(y,x);
                    % Other (failed) things I tried
%                     b(e) = im_background(y,x,channel)-im_s(y,x,channel);
%                     b(e) = im_s(y,x,channel);
%                     b(e) = 5*(im_s(y,x,channel)-im_background(y,x,channel));
                else
                    b(e) = im_background(y,x,channel);
                    Ay(d) = e; Ax(d) = e; Avalue(d) = 1;
                    d=d+1;
                end
            end
        end
%         Ax(Ax==0) = []; Ay(Ay==0) = []; Avalue(Avalue==0) = [];
        A = sparse(Ay,Ax,Avalue,length(b),length(b));
        v = lscov(A,b);
        im_out(:,:,channel) = reshape(full(v),h,w);
    end
end