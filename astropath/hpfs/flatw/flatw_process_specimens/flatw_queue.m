function flatw_queue(main)
%
i1 = 1;
%
%main = '\\bki05\Processing_Specimens';
%
while i1 == 1
    %
    try
        process_flt2bin(main)
        process_flatw_queue(main)
        Send_process(main)
    catch
    end
    %
    pause(30*60)
    %
end
%
end