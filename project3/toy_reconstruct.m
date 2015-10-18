function im_out = toy_reconstruct(toyim)
    [h, w, ~] = size(toyim); 
    im2var = zeros(h, w); 
    im2var(1:h*w) = 1:h*w;
    
    A = zeros(2*(h-1)*(w-1)+1, h*w);
    b = zeros(2*(h-1)*(w-1)+1,1);
    e=1;
    for x=1:w-1
        for y=1:h-1
            A(e, im2var(y,x+1))=1;
            A(e, im2var(y,x))=-1;
            b(e) = toyim(y,x+1)-toyim(y,x);
            e=e+1;
            A(e, im2var(y+1,x))=1;
            A(e, im2var(y,x))=-1;
            b(e) = toyim(y+1,x)-toyim(y,x);
            e=e+1;
        end
    end
    A(e,im2var(1,1))=1;
    b(e)=toyim(1,1);
    A = sparse(A);
    
    v = lscov(A,b);
    
    im_out = reshape(full(v),h,w);
end