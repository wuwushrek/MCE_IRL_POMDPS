pomdp

observables
start
endobservables

const int N=10;
const int axMAX = N-1;
const int ayMAX = N-1;
const int axMIN = 0;
const int ayMIN = 0;
const double slippery = 0.1;
const int ob1x = axMAX-1;
const int ob1y = ayMAX-1;
const int ob2x = axMAX;
const int ob2y = ayMIN+1;
const int ob3x = axMIN+1;
const int ob3y = ayMIN;
const int ob4x = axMAX;
const int ob4y = ayMAX-1;
const int ob5x = axMAX-3;
const int ob5y = ayMAX-1;

formula done = start & ax = axMAX & ay = ayMAX;
observable "amdone" = done;
formula crash =  (ax = ob1x & ay = ob1y) | (ax = ob2x & ay = ob2y)  | (ax = ob3x & ay = ob3y) | (ax = ob4x & ay = ob4y) | (ax = ob5x & ay = ob5y)  ;
observable "hascrash" = crash;
observable "finished" = finished;


module master
    start : bool init false;
    finished : bool init false;

    [placement] !start -> (start'=true);
    [north] start & !finished -> true;
    [south] start  & !finished -> true;
    [east] start  & !finished-> true;
    [west] start & !finished -> true;
    [finish] !finished & done -> (finished'=true);


endmodule


module robot
    ax : [axMIN..axMAX] init 0;
    ay : [ayMIN..ayMAX] init 0;
    slipped : bool init false;

    [placement] !start ->  1/4: (ax'=ob1x-1) & (ay'=ob1y) + 1/4: (ax'=1) & (ay'=1) + 1/4: (ax'=2) & (ay'=1) + 1/4: (ax'=1) & (ay'=3);

    [west] true -> (1-slippery): (ax'=max(ax-1,axMIN)) + slippery: (ax'=max(ax-2,axMIN));
    [east] true -> (1-slippery): (ax'=min(ax+1,axMAX)) + slippery: (ax'=min(ax+2,axMAX));
    [south]  true -> (1-slippery): (ay'=min(ay+1,ayMAX)) + slippery: (ay'=min(ay+2,ayMAX));
    [north]  true -> (1-slippery): (ay'=max(ay-1,ayMIN)) + slippery: (ay'=max(ay-2,ayMIN));
endmodule

rewards "crash_state"
    [west] crash : -1;
    [east] crash : -1;
    [south] crash : -1;
    [north] crash : -1;
endrewards

rewards "time"
    [west] true : -1;
    [east] true : -1;
    [south] true : -1;
    [north] true : -1;
endrewards

rewards "finish"
    [finish] done : 1;
endrewards

label "goal" = finished & done;
label "traps" = crash;
label "notbad" =  !crash;