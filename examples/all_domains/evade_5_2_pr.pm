pomdp

observables
start, dx, dy, turn
endobservables

const int RADIUS=2;

const int N=5;
const int xMAX = N-1;
const int yMAX = N-1;
const int aXMAX = xMAX;
const int aYMAX = yMAX;
const int aXMIN = 0;
const int aYMIN = 1;
const int dxMAX = xMAX;
const int dyMAX = yMAX;
const int dxMIN = 0;
const int dyMIN = 0;
const double slippery = 0;

formula done = start & dx = dxMAX & dy = dyMAX;
observable "amdone" = done;
formula crash =  (dx = ax & dy = ay);
observable "hascrash" = crash;
formula seedrone = ax-dx < RADIUS + 1 & dx-ax < RADIUS +1 & ay-dy < RADIUS+1 & dy -ay < RADIUS+1;
observable "seedx" = (seedrone |justscanned) ? ax : -1;
observable "seedy" = (seedrone |justscanned)  ? ay : -1;
observable "amfinish" = finished;


module master
    start : bool init false;
    turn : bool init false;
    justscanned : bool init false;
    finished : bool init false;

    [placement] !start -> (start'=true);
    [north] start & !finished & turn -> (turn'=!turn);
    [south] start & !finished & turn -> (turn'=!turn);
    [east]  start & !finished & turn -> (turn'=!turn);
    [west]  start & !finished & turn -> (turn'=!turn);
    [adv]  start & !finished & !turn -> (turn'=!turn) & (justscanned'=false);
    [scan] start & !finished & turn -> (justscanned'=true);
    [finish] !finished & done -> (finished'=true);
endmodule


module drone
    dx : [dxMIN..dxMAX] init 0;
    dy : [dyMIN..dyMAX] init 0;

    [west] true -> (1-slippery): (dx'=max(dx-1,dxMIN)) + slippery: (dx'=max(dx,dxMIN));
    [east] true -> (1-slippery): (dx'=min(dx+1,dxMAX)) + slippery: (dx'=min(dx,dxMAX));
    [south]  true -> (1-slippery): (dy'=min(dy+1,dyMAX)) + slippery: (dy'=min(dy,dyMAX));
    [north]  true -> (1-slippery): (dy'=max(dy-1,dyMIN)) + slippery: (dy'=max(dy,dyMIN));
    [scan] true -> 1:(dx'=dx);
endmodule



module agent
    ax : [aXMIN..aXMAX] init aXMAX-1;
    ay : [aYMIN..aYMAX] init aYMAX;

    [adv] true -> 1/8 : (ax'=max(ax-1,aXMIN)) +  1/8: (ax'=min(ax+1,aXMAX))
                + 1/8 : (ay'=max(ay-1,aYMIN)) + 1/8 : (ay'=min(ay+1,aYMAX))
                + 1/16 : (ax'=max(ax-2,aXMIN)) +  1/16: (ax'=min(ax+2,aXMAX))
                + 1/16 : (ay'=max(ay-2,aYMIN)) + 1/16 : (ay'=min(ay+2,aYMAX))
                + 1/16 : (ax'=max(ax-1,aXMIN)) & (ay'=max(ay-1,aYMIN)) +  1/16: (ax'=min(ax+1,aXMAX)) & (ay'=max(ay-1,aYMIN))
                + 1/16 : (ax'=max(ax-1,aXMIN)) & (ay'=min(ay+1,aYMAX)) +  1/16: (ax'=min(ax+1,aXMAX)) & (ay'=min(ay+1,aYMAX));
endmodule

rewards "crash_state"
    [south] crash : 10;
    [north] crash : 10;
    [west] crash : 10;
    [east] crash : 10;
    [scan] crash : 10;
    [adv] crash : 10;
endrewards

rewards "finish"
    [finish] done : 100;
endrewards

label "goal" = done;
label "traps" = crash;
label "notbad" =  !crash;