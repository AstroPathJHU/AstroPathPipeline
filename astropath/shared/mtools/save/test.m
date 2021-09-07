function test(Q,T)
    k=0;
    for i=1:numel(T.cx)
        if(T.cx(i) == Q.cx(i+k) & T.cy(i)==Q.cy(i+k))
            continue
        else
            i
            T(i,:)
            Q(i,:)
            k = k+1;
        end
    end
end