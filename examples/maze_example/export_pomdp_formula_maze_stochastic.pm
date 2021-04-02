// Exported by storm
// Original model type: POMDP
@type: POMDP
@parameters

@reward_models
total_time poisonous goal_reach 
@nr_states
16
@nr_choices
55
@model
state 0 {3} [0, 0, 0] init
//[s=-1	& o=0]
	action __NOLABEL__ [0, 0, 0]
		1 : 0.09090909091
		2 : 0.09090909091
		3 : 0.09090909091
		4 : 0.09090909091
		5 : 0.09090909091
		6 : 0.09090909091
		7 : 0.09090909091
		8 : 0.09090909091
		9 : 0.09090909091
		10 : 0.09090909091
		11 : 0.09090909091
state 1 {7} [0, 0, 0]
//[s=0	& o=1]
	action east [-1, 0, 0]
		1 : 0.1
		2 : 0.8
		6 : 0.1
	action west [-1, 0, 0]
		1 : 0.8
		2 : 0.1
		6 : 0.1
	action north [-1, 0, 0]
		1 : 0.8
		2 : 0.1
		6 : 0.1
	action south [-1, 0, 0]
		1 : 0.1
		2 : 0.1
		6 : 0.8
state 2 {1} [0, 0, 0]
//[s=1	& o=2]
	action east [-1, 0, 0]
		1 : 0.1
		2 : 0.1
		3 : 0.8
	action west [-1, 0, 0]
		1 : 0.8
		2 : 0.1
		3 : 0.1
	action north [-1, 0, 0]
		1 : 0.1
		2 : 0.8
		3 : 0.1
	action south [-1, 0, 0]
		1 : 0.1
		2 : 0.8
		3 : 0.1
state 3 {8} [0, 0, 0]
//[s=2	& o=3]
	action east [-1, 0, 0]
		2 : 0.1
		3 : 0.1
		4 : 0.7
		7 : 0.1
	action west [-1, 0, 0]
		2 : 0.7
		3 : 0.1
		4 : 0.1
		7 : 0.1
	action north [-1, 0, 0]
		2 : 0.1
		3 : 0.7
		4 : 0.1
		7 : 0.1
	action south [-1, 0, 0]
		2 : 0.1
		3 : 0.1
		4 : 0.1
		7 : 0.7
state 4 {1} [0, 0, 0]
//[s=3	& o=2]
	action east [-1, 0, 0]
		3 : 0.1
		4 : 0.1
		5 : 0.8
	action west [-1, 0, 0]
		3 : 0.8
		4 : 0.1
		5 : 0.1
	action north [-1, 0, 0]
		3 : 0.1
		4 : 0.8
		5 : 0.1
	action south [-1, 0, 0]
		3 : 0.1
		4 : 0.8
		5 : 0.1
state 5 {2} [0, 0, 0]
//[s=4	& o=4]
	action east [-1, 0, 0]
		4 : 0.1
		5 : 0.8
		8 : 0.1
	action west [-1, 0, 0]
		4 : 0.8
		5 : 0.1
		8 : 0.1
	action north [-1, 0, 0]
		4 : 0.1
		5 : 0.8
		8 : 0.1
	action south [-1, 0, 0]
		4 : 0.1
		5 : 0.1
		8 : 0.8
state 6 {0} [0, 0, 0]
//[s=5	& o=5]
	action east [-1, 0, 0]
		1 : 0.1
		6 : 0.8
		9 : 0.1
	action west [-1, 0, 0]
		1 : 0.1
		6 : 0.8
		9 : 0.1
	action north [-1, 0, 0]
		1 : 0.8
		6 : 0.1
		9 : 0.1
	action south [-1, 0, 0]
		1 : 0.1
		6 : 0.1
		9 : 0.8
state 7 {0} [0, 0, 0]
//[s=6	& o=5]
	action east [-1, 0, 0]
		3 : 0.1
		7 : 0.8
		10 : 0.1
	action west [-1, 0, 0]
		3 : 0.1
		7 : 0.8
		10 : 0.1
	action north [-1, 0, 0]
		3 : 0.8
		7 : 0.1
		10 : 0.1
	action south [-1, 0, 0]
		3 : 0.1
		7 : 0.1
		10 : 0.8
state 8 {0} [0, 0, 0]
//[s=7	& o=5]
	action east [-1, 0, 0]
		5 : 0.1
		8 : 0.8
		11 : 0.1
	action west [-1, 0, 0]
		5 : 0.1
		8 : 0.8
		11 : 0.1
	action north [-1, 0, 0]
		5 : 0.8
		8 : 0.1
		11 : 0.1
	action south [-1, 0, 0]
		5 : 0.1
		8 : 0.1
		11 : 0.8
state 9 {0} [0, 0, 0]
//[s=8	& o=5]
	action east [-1, 0, 0]
		6 : 0.1
		9 : 0.8
		12 : 0.1
	action west [-1, 0, 0]
		6 : 0.1
		9 : 0.8
		12 : 0.1
	action north [-1, 0, 0]
		6 : 0.8
		9 : 0.1
		12 : 0.1
	action south [-1, 0, 0]
		6 : 0.1
		9 : 0.1
		12 : 0.8
state 10 {0} [0, 0, 0]
//[s=9	& o=5]
	action east [-1, 0, 0]
		7 : 0.1
		10 : 0.8
		13 : 0.1
	action west [-1, 0, 0]
		7 : 0.1
		10 : 0.8
		13 : 0.1
	action north [-1, 0, 0]
		7 : 0.8
		10 : 0.1
		13 : 0.1
	action south [-1, 0, 0]
		7 : 0.1
		10 : 0.1
		13 : 0.8
state 11 {0} [0, 0, 0]
//[s=10	& o=5]
	action east [-1, 0, 0]
		8 : 0.1
		11 : 0.8
		14 : 0.1
	action west [-1, 0, 0]
		8 : 0.1
		11 : 0.8
		14 : 0.1
	action north [-1, 0, 0]
		8 : 0.8
		11 : 0.1
		14 : 0.1
	action south [-1, 0, 0]
		8 : 0.1
		11 : 0.1
		14 : 0.8
state 12 {5} [0, 0, 0]
//[s=11	& o=6]
	action done [0, 0, 0]
		12 : 1
state 13 {9} [0, 0, 0]
//[s=13	& o=7]
	action east [-1, 0, 1]
		15 : 1
	action west [-1, 0, 1]
		15 : 1
	action north [-1, 0, 1]
		15 : 1
	action south [-1, 0, 1]
		15 : 1
state 14 {4} [0, 0, 0] poison_light
//[s=12	& o=8]
	action east [-1, -1, 0]
		14 : 1
	action west [-1, -1, 0]
		14 : 1
	action north [-1, -1, 0]
		11 : 0.9
		14 : 0.1
	action south [-1, -1, 0]
		14 : 1
state 15 {6} [0, 0, 0] target
//[s=14	& o=9]
	action done [0, 0, 0]
		15 : 1
