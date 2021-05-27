pomdp

observables
start, dx, dy, turn
endobservables

const int RADIUS=2;

const int N=10;
const int xMAX = N-1;
const int yMAX = N-1;
const int axMAX = xMAX;
const int ayMAX = yMAX;
const int axMIN = 0;
const int ayMIN = 0;
const int dxMAX = xMAX;
const int dyMAX = yMAX;
const int dxMIN = 0;
const int dyMIN = 0;
const double slippery = 0;
const bool DIAGONAL = false;
const int CAMERAXMIN = 0;
const int CAMERAYMIN = floor((ayMIN + ayMAX)/2);
const int CAMERAXMAX = xMAX;
const int CAMERAYMAX = CAMERAYMIN + 1;



formula northenabled = dx != dxMIN;
formula southenabled = dx != dxMAX;
formula westenabled = dy != dyMIN;
formula eastenabled = dy != dyMAX;
formula done = start & (dx = ax & dy = ay);
observable "amdone" = done;
formula left =  (ax = axMIN & ay = ayMAX-1) | (ax = axMIN + 1 & ay = ayMIN)  ;
observable "hasleft" = left;
formula laserdet = ax >= CAMERAXMIN & ax <= CAMERAXMAX & ay >= CAMERAYMIN & ay <= CAMERAYMAX;
formula seedrone = ax-dx < RADIUS + 1 & dx-ax < RADIUS +1 & ay-dy < RADIUS+1 & dy -ay < RADIUS+1;
observable "seedx" = (laserdet | seedrone) ? ax : -1;
observable "seedy" = (laserdet | seedrone) ? ay : -1;
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
    dx : [dxMIN..dxMAX] init 0;
    dy : [dyMIN..dyMAX] init 3;

    [west] true -> (1-slippery): (dx'=max(dx-1,dxMIN)) + slippery: (dx'=max(dx,dxMIN));
    [east] true -> (1-slippery): (dx'=min(dx+1,dxMAX)) + slippery: (dx'=min(dx,dxMAX));
    [south]  true -> (1-slippery): (dy'=min(dy+1,dyMAX)) + slippery: (dy'=min(dy,dyMAX));
    [north]  true -> (1-slippery): (dy'=max(dy-1,dyMIN)) + slippery: (dy'=max(dy,dyMIN));
endmodule



module agent
    ax : [axMIN..axMAX] init axMAX-1;
    ay : [ayMIN..ayMAX] init ayMAX;

    [adv] !DIAGONAL -> 1/4 : (ax'=max(ax-1,axMIN)) +  1/4 : (ax'=min(ax+1,axMAX)) + 1/4 : (ay'=max(ay-1,ayMIN)) + 1/4 : (ay'=min(ay+1,ayMAX));
    [adv] DIAGONAL -> 1/8 : (ax'=max(ax-1,axMIN)) +  1/8: (ax'=min(ax+1,axMAX))
                + 1/8 : (ay'=max(ay-1,ayMIN)) + 1/8 : (ay'=min(ay+1,ayMAX))
                + 1/8 : (ax'=max(ax-1,axMIN)) & (ay'=max(ay-1,ayMIN)) +  1/8: (ax'=min(ax+1,axMAX)) & (ay'=max(ay-1,ayMIN))
                + 1/8 : (ax'=max(ax-1,axMIN)) & (ay'=min(ay+1,ayMAX)) +  1/8: (ax'=min(ax+1,axMAX)) & (ay'=min(ay+1,ayMAX));
endmodule

rewards "left"
    [] left : -1;
endrewards

rewards "finish"
    [finish] done : 1;
endrewards


label "goal" = done;
label "notbad" =  !left;
label "exits" = left;