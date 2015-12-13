# This uses value iteration to create the best policy for the small data set
# 1) Use maximum likelihood estimation to learn the Transition and Reward functions from the given data
# 2) Implement value iteration to calculate the utility of each state
# 3) Extract the best policy (action that maximizes utility at each state)

using DataFrames
using DataStructures

#TUNING PARAMETERS
discountFactor = 0.9
numIterations = 10

#Defining Data Set
DATA_SET = "small"
outputFile = open(string(DATA_SET,".policy"),"w")
NUM_STATES_X = 10
NUM_STATES_Y = 10

data = readtable(string(DATA_SET,".csv"));

# Initialize counters as dicts
s_a_counter = Dict();   #(state, action)
s_a_sp_counter = Dict();   #(state, action, state')
rewards_counter = Dict();   #(state, action);


# Read data into counters
# for SMALL data set
state_set = Set(); # state = (row, col)
action_array = [1,2,3,4,5]
for i = 1:1:size(data,1)
    state = (data[i,1], data[i,2])
    if(!in(state, state_set))
        push!(state_set,state)
    end
    s_a = (data[i,1], data[i,2], data[i,3])
    s_a_sp = (data[i,1], data[i,2], data[i,3], data[i,4], data[i,5])
    s_a_counter[s_a] = get(s_a_counter,s_a,0) + 1
    s_a_sp_counter[s_a_sp] = get(s_a_sp_counter,s_a_sp,0) + 1
    rewards_counter[s_a] = get(rewards_counter,s_a,0) + data[i,6]
end

TransitionMap = Dict();
for (s_a_sp, count) in s_a_sp_counter
    s_a = s_a_sp[1:3]
    TransitionMap[s_a_sp] = count/s_a_counter[s_a]
end

RewardsMap = Dict();
for (s_a, reward) in rewards_counter
    RewardsMap[s_a] = reward/s_a_counter[s_a]
end

#Value Iteration - Gauss-Seidel (update utility values in place)
UtilityMap = Dict(); # state => (maxUtility, action for this utility)
for i=1:1:numIterations
    for state in state_set
        max_s_a_util = 0  #max utility for this state
        max_action = 0  #best action for this state
        for action in action_array
            sum_successor_util = 0
            #for all successor states, add the transition prob * utility
            for succ_state in state_set
                s_a_sp = (state[1],state[2],action,succ_state[1],succ_state[2])
                sum_successor_util += get(TransitionMap,s_a_sp,(0,0))[1]*get(UtilityMap,succ_state,(0,0))[1]  #note: getting utility of successor state, not current state
            end
            s_a_util = RewardsMap[(state[1],state[2],action)] + discountFactor*sum_successor_util
            if s_a_util > max_s_a_util
                max_s_a_util = s_a_util
                max_action = action
            end
        end
        UtilityMap[state] = (max_s_a_util, max_action)
    end
end

#Extract best policy

BestPolicy = Dict();
start_of_file = true;
num_no_action_found = 0;
for y=1:1:10
    for x=1:1:10
        state = (x,y)
        action = UtilityMap[state][2]
        if (action < 0) #no action found
            num_no_action_found += 1
            action = rand(action_array)  #pick random action
        end
        if !start_of_file
            write(outputFile, string("\n", action))
        else
            write(outputFile, string(action))  # makes sure that we have no trailing empty lines in output
            start_of_file = false
        end
    end
end

close(outputFile)


println("---- Printing Results ----")
println("Num States with no best action policy found: ", num_no_action_found)
action_found_rate = (NUM_STATES_X*NUM_STATES_Y-num_no_action_found)/(NUM_STATES_X*NUM_STATES_Y)
println("Rate of best action found: ", action_found_rate*100, "%")

