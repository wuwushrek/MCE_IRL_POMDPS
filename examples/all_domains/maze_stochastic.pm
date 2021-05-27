// maze example (POMDP)
// slightly extends that presented in
// Littman, Cassandra and Kaelbling
// Learning policies for partially observable environments: Scaling up  
// Technical Report CS, Brown University
// gxn 29/01/16

// state space (value of variable "s")

//  0  1  2  3  4
//  5     6     7
//  8     9    10
// 11     13   12

// 13 is the target

pomdp

// can observe the walls and target
observables
	o
endobservables
// o=0 - observation in initial state
// o=1 - west and north walls (s0)
// o=2 - north and south walls (s1 and s3)
// o=3 - north wall (s2)
// o=4 - east and north way (s4)
// o=5 - east and west walls (s5, s6, s7, s8, s9 and s10)
// o=6 - east, west and south walls (s11 and s12)
// o=7 - the target (s13)


const double p=0.1;

module maze

	s : [-1..15];
	o : [0..10];
	
	// initialisation
	[] s=-1 -> 1/11 : (s'=0) & (o'=1)
			 + 1/11 : (s'=1) & (o'=2)
			 + 1/11 : (s'=2) & (o'=3)
			 + 1/11 : (s'=3) & (o'=2)
			 + 1/11 : (s'=4) & (o'=4)
			 + 1/11 : (s'=5) & (o'=5)
			 + 1/11 : (s'=6) & (o'=5)
			 + 1/11 : (s'=7) & (o'=5)
			 + 1/11 : (s'=8) & (o'=5)
			 + 1/11 : (s'=9) & (o'=5)
			 + 1/11 : (s'=10) & (o'=5);
	
	// moving around the maze
	
	[east] s=0 -> (1-2*p): (s'=1) & (o'=2) + p:(s'=0) + p:(s'=5) & (o'=5);
	[west] s=0 -> (1-2*p):(s'=0) +p: (s'=1) & (o'=2) + p:(s'=5) & (o'=5);
	[north] s=0 -> (1-2*p):(s'=0) +p: (s'=1) & (o'=2) + p:(s'=5) & (o'=5);
	[south] s=0 -> p:(s'=0) +p: (s'=1) & (o'=2) + (1-2*p):(s'=5) & (o'=5);

	[east] s=1 ->  (1-2*p) : (s'=2) & (o'=3) + p : (s'=0) & (o'=1) + p : (s'=1);
	[west] s=1 ->  p : (s'=2) & (o'=3) + (1-2*p) : (s'=0) & (o'=1) + p : (s'=1);
	[north] s=1 -> p : (s'=2) & (o'=3) + p : (s'=0) & (o'=1) + (1-2*p) : (s'=1);
	[south] s=1 -> p : (s'=2) & (o'=3) + p : (s'=0) & (o'=1) + (1-2*p) : (s'=1);

	[east] s=2 ->  (1-3*p): (s'=3) & (o'=2) + p : (s'=1) & (o'=2) + p : (s'=2) + p : (s'=6) & (o'=5);
	[west] s=2 ->  p: (s'=3) & (o'=2) + (1-3*p) : (s'=1) & (o'=2) + p : (s'=2) + p : (s'=6) & (o'=5);
	[north] s=2 -> p: (s'=3) & (o'=2) + p : (s'=1) & (o'=2) + (1-3*p) : (s'=2) + p : (s'=6) & (o'=5);
	[south] s=2 -> p: (s'=3) & (o'=2) + p : (s'=1) & (o'=2) + p : (s'=2) + (1-3*p) : (s'=6) & (o'=5);

	[east] s=3 ->  (1-2*p) : (s'=4) & (o'=4) + p : (s'=2) & (o'=3) + p : (s'=3);
	[west] s=3 ->  p : (s'=4) & (o'=4) + (1-2*p) : (s'=2) & (o'=3) + p : (s'=3);
	[north] s=3 -> p : (s'=4) & (o'=4) + p : (s'=2) & (o'=3) + (1-2*p) : (s'=3);
	[south] s=3 -> p : (s'=4) & (o'=4) + p : (s'=2) & (o'=3) + (1-2*p) : (s'=3);

	[east] s=4 -> (1-2*p) : (s'=4) + p : (s'=3) & (o'=2) + p : (s'=7) & (o'=5);
	[west] s=4 -> p : (s'=4) + (1-2*p) : (s'=3) & (o'=2) + p : (s'=7) & (o'=5);
	[north] s=4 -> (1-2*p) : (s'=4) + p : (s'=3) & (o'=2) + p : (s'=7) & (o'=5);
	[south] s=4 -> p : (s'=4) + p : (s'=3) & (o'=2) + (1-2*p) : (s'=7) & (o'=5);

	[east] s=5 ->  (1-2*p) : (s'=5) + p : (s'=0) & (o'=1) + p : (s'=8);
	[west] s=5 ->  (1-2*p) : (s'=5) + p : (s'=0) & (o'=1) + p : (s'=8);
	[north] s=5 -> p : (s'=5) + (1-2*p) : (s'=0) & (o'=1) + p : (s'=8);
	[south] s=5 -> p : (s'=5) + p : (s'=0) & (o'=1) + (1-2*p) : (s'=8);

	[east] s=6 ->  (1-2*p) : (s'=6) + p : (s'=2) & (o'=3) + p : (s'=9);
	[west] s=6 ->  (1-2*p) : (s'=6) + p : (s'=2) & (o'=3) + p : (s'=9);
	[north] s=6 -> p : (s'=6) + (1-2*p) : (s'=2) & (o'=3) + p : (s'=9);
	[south] s=6 -> p : (s'=6) + p : (s'=2) & (o'=3) + (1-2*p) : (s'=9);

	[east] s=7 ->  (1-2*p) : (s'=7) + p : (s'=4) & (o'=4) + p : (s'=10);
	[west] s=7 ->  (1-2*p) : (s'=7) + p : (s'=4) & (o'=4) + p : (s'=10);
	[north] s=7 -> p : (s'=7) + (1-2*p) : (s'=4) & (o'=4) + p : (s'=10);
	[south] s=7 -> p : (s'=7) + p : (s'=4) & (o'=4) + (1-2*p) : (s'=10);

	[east] s=8 ->  (1-2*p) : (s'=8) + p : (s'=5) + p : (s'=11) & (o'=6);
	[west] s=8 ->  (1-2*p) : (s'=8) + p : (s'=5) + p : (s'=11) & (o'=6);
	[north] s=8 -> p : (s'=8) + (1-2*p) : (s'=5) + p : (s'=11) & (o'=6);
	[south] s=8 -> p : (s'=8) + p : (s'=5) + (1-2*p) : (s'=11) & (o'=6);

	[east] s=9 ->  (1-2*p) : (s'=9) + p : (s'=6) + p : (s'=13) & (o'=7);
	[west] s=9 ->  (1-2*p) : (s'=9) + p : (s'=6) + p : (s'=13) & (o'=7);
	[north] s=9 -> p : (s'=9) + (1-2*p) : (s'=6) + p : (s'=13) & (o'=7);
	[south] s=9 -> p : (s'=9) + p : (s'=6) + (1-2*p) : (s'=13) & (o'=7);

	[east] s=10 ->  (1-2*p) : (s'=10) + p : (s'=7) + p : (s'=12) & (o'=8);
	[west] s=10 ->  (1-2*p) : (s'=10) + p : (s'=7) + p : (s'=12) & (o'=8);
	[north] s=10 -> p : (s'=10) + (1-2*p) : (s'=7) + p : (s'=12) & (o'=8);
	[south] s=10 -> p : (s'=10) + p : (s'=7) + (1-2*p) : (s'=12) & (o'=8);
	//loop when we reach trap poison
	[east] s=11 -> (s'=15) & (o'=10);
	[west] s=11 -> (s'=15) & (o'=10);
	[north] s=11 -> (s'=15) & (o'=10);
	[south] s=11 -> (s'=15) & (o'=10);
	[done] s=15 -> (s'=15);


	[east] s=12 -> (s'=12) & (o'=8);
	[west] s=12 -> (s'=12) & (o'=8);
	[north] s=12 -> (1-p): (s'=10) & (o'=5) + p: (s'=12) & (o'=8);
	[south] s=12 -> (s'=12) & (o'=8);

	// loop when we reach the target
	[east] s=13 -> (s'=14) & (o'=9);
	[west] s=13 -> (s'=14) & (o'=9);
	[north] s=13 -> (s'=14) & (o'=9);
	[south] s=13 -> (s'=14) & (o'=9);

	[done] s=14 -> (s'=14);
endmodule

// First reward feature -> time to reach the target
rewards "total_time"
	[east] true : -1;
	[west] true : -1;
	[north] true : -1;
	[south] true : -1;
endrewards

// Second reward feature -> weight when getting in a poisonous states
rewards "poisonous"
	[east] o = 8 : -1;
	[west] o = 8 : -1;
	[north] o = 8 : -1;
	[south] o = 8 : -1;
	[east] o = 6 : -1;
	[west] o = 6 : -1;
	[north] o = 6 : -1;
	[south] o = 6 : -1;
endrewards

// Third reward feature -> reach the goal
rewards "goal_reach"
	[east] o = 7 : 1;
	[west] o = 7 : 1;
	[north] o = 7 : 1;
	[south] o = 7 : 1;
endrewards

// target observation
label "target" = o=9;

// trap poison
label "bad_poison" = o=10;

// Poisonous states
label "poison_light" = o=8;
