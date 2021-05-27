pomdp

observables
start, dx, dy, turn
endobservables


const int RADIUS=2;
const int N=4;
const int ARADIUS = 1;
const int ayPOS = 2;
const int ay2POS = N-2;
const int xMAX = N-1;
const int yMAX = N-1;
const int axMAX = xMAX;
const int axMIN = 0;
const int dxMAX = xMAX;
const int dyMAX = yMAX;
const int dxMIN = 0;
const int dyMIN = 0;
const double slippery = 0.1;


formula westenabled = dx != dxMIN;
formula eastenabled = dx != dxMAX;
formula northenabled = dy != dyMIN;
formula southenabled = dy != dyMAX;
formula done = start & dx = dxMAX & dy = dyMAX;
observable "amdone" = done;
formula crash =  (dx = ax & dy = ay);
observable "hascrash" = crash;
formula seedrone = ax-dx < RADIUS + 1 & dx-ax < RADIUS +1 & ay-dy < RADIUS+1 & dy -ay < RADIUS+1;
observable "seedx" = seedrone ? ax : -1;
observable "seedy" = seedrone ? ay : -1;
observable "dir" = seedrone ? dir : -1;
formula seedrone2 = ax2-dx < RADIUS + 1 & dx-ax2 < RADIUS +1 & ay2-dy < RADIUS+1 & dy -ay2 < RADIUS+1;
observable "seedx2" = seedrone2 ? ax2 : -1;
observable "seedy2" = seedrone2 ? ay2 : -1;
observable "dir2" = seedrone2 ? dir2 : -1;

formula seeagent1 = ax-dx < ARADIUS + 1 & dx-ax < ARADIUS + 1 & ay-dy < ARADIUS+1 & dy -ay < ARADIUS+1;
formula seeagent2 = ax2-dx < ARADIUS + 1 & dx-ax2 < ARADIUS + 1 & ay2-dy < ARADIUS+1 & dy -ay2 < ARADIUS+1;

observable "seen" = (seeagent1 | seeagent2);
observable "finished" = finished;

module master
    start : bool init false;
    turn : bool init false;
    finished : bool init false;

    [placement] !start -> (start'=true);
    [north] start & turn -> (turn'=!turn);
    [south] start & !finished & turn -> (turn'=!turn);
    [east]  start & !finished & turn -> (turn'=!turn);
    [west]  start & !finished & turn -> (turn'=!turn);
    [adv]  start & !finished & !turn -> (turn'=!turn);
    [finish] !finished & done -> (finished'=true);
endmodule


module drone
    dx : [dxMIN..dxMAX] init dxMIN;
    dy : [dyMIN..dyMAX] init dyMIN;

    [west] westenabled -> (1-slippery): (dx'=max(dx-1,dxMIN)) + slippery: (dx'=max(dx,dxMIN));
    [east] eastenabled -> (1-slippery): (dx'=min(dx+1,dxMAX)) + slippery: (dx'=min(dx,dxMAX));
    [south]  southenabled -> (1-slippery): (dy'=min(dy+1,dyMAX)) + slippery: (dy'=min(dy,dyMAX));
    [north]  northenabled -> (1-slippery): (dy'=max(dy-1,dyMIN)) + slippery: (dy'=max(dy,dyMIN));
endmodule


module agent
    ax : [axMIN..axMAX] init axMAX;
    ay : [ayPOS..ayPOS+1] init ayPOS;
    dir : [0..1] init 1;

    [placement] true -> 1: (ay'=ayPOS);// + 0.5: (ay'=ayPOS+1);
    [adv] dir=0 & ax < axMAX -> 0.5: (ax'=min(axMAX,ax+1)) +  0.5: (ax'=min(axMAX,ax+2));
    [adv] dir=1 & ax > axMIN -> 0.5: (ax'=max(axMIN,ax-1)) +  0.5: (ax'=max(axMIN,ax-2));
    [adv] ax = axMAX & dir=0 -> (dir'=1-dir);
    [adv] ax = axMIN & dir=1 -> (dir'=1-dir);
endmodule


module agent2=agent[ax=ax2,ay=ay2,dir=dir2,ayPOS=ay2POS] endmodule

rewards "crash_state"
    [] crash : -1;
endrewards

rewards "finish"
    [finish] done : 1;
endrewards

rewards "avoid"
    [] !crash & !seeagent1 & !seeagent2 : 1;
endrewards


label "goal" = done;
label "traps" = crash;
label "notbad" =  !crash & !seeagent1 & !seeagent2;