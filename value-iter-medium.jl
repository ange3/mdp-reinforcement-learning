# This uses value iteration to create the best policy for the medium data set
# 1) Use maximum likelihood estimation to learn the Transition and Reward functions from the given data
# 2) Implement value iteration to calculate the utility of each state
# 3) Extract the best policy (action that maximizes utility at each state)

using DataFrames
using DataStructures

#TUNING PARAMETERS
discountFactor = 0.99
numIterations = 10
println("Number of Iterations ", numIterations)


#Defining Data Set
DATA_SET = "medium"
outputFile = open(string(DATA_SET,".policy"),"w")
NUM_STATES_OMEGA = 501
NUM_STATES_THETA = 501
theta_buckets_divisor = 0.01256 #to create 501 evenly spaced buckets from 0 to 2 pi, inclusive
omega_buckets_divisor = 0.04 #to create 501 evenly spaced buckets from -10 to 10, inclusive
action_array = [0,10]
NUM_ACTIONS = size(action_array)[1]

#Read data
data = readtable(string(DATA_SET,".csv"));
# Initialize counters
s_a_counter = Dict();   #(state, action) --> count
s_a_sp_counter = Dict();   #(state, action, state') --> count
rewards_counter = Dict();   #(state, action) --> reward
successor_state_map = Dict();  #(state) --> [list of successor states]

# Read data into counters
# for MEDIUM data set - discretizing theta and omega values for each state
state_set = Set();  # state = (theta, omega)
for i = 1:1:size(data,1)
    theta = round(data[i,1]/theta_buckets_divisor)
    omega = round(data[i,2]/omega_buckets_divisor)
    theta_p = round(data[i,4]/theta_buckets_divisor)
    omega_p = round(data[i,5]/omega_buckets_divisor)
    state = (theta, omega)
    state_p = (theta_p, omega_p)
    if(!in(state, state_set))
        push!(state_set,state)
    end
    #append new successor state to state in map
    if(haskey(successor_state_map, state))
        successor_state_map[state] = append!(successor_state_map[state], [state_p]) 
    else
        successor_state_map[state] = [state_p]
    end
    s_a = (theta, omega, data[i,3])
    s_a_sp = (theta, omega, data[i,3], theta_p, omega_p)
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

println("Running Value Iteration!")

#Value Iteration - Gauss-Seidel (update utility values in place)
UtilityMap = Dict(); # state => (maxUtility, action for this utility)
for i=1:1:numIterations
    println("**Iteration ", i)
    for state in state_set
        max_s_a_util = 0  #max utility for this state
        max_action = 0  #best action for this state
        numTimesUpdated = get(UtilityMap,state,(0,0,0))[3] #number of times this state's utility has been updated
        for action in action_array
            sum_successor_util = 0
            #for all successor states, add the transition prob * utility
            for succ_state in successor_state_map[state]
                s_a_sp = (state[1],state[2],action,succ_state[1],succ_state[2])
                sum_successor_util += get(TransitionMap,s_a_sp,(0,0))[1]*get(UtilityMap,succ_state,(0,0,0))[1]  #note: getting utility of successor state, not current state
            end
            s_a_util = get(RewardsMap,(state[1],state[2],action),0) + discountFactor*sum_successor_util
            # COUNT! can count here on rewardsMap if we've never seen this s,a pair before
            if s_a_util > max_s_a_util
                #println("for state ", state, ": greater max! NEW: ", s_a_util," OLD MAX: ", max_s_a_util)
                max_s_a_util = s_a_util
                max_action = action
                numTimesUpdated +=1
            end
        end
        UtilityMap[state] = (max_s_a_util, max_action, numTimesUpdated)
    end
end

#Extract best policy
println("Extracting best policy!")

BestPolicy = Dict();
start_of_file = true;
num_no_action_found = 0;
for y=round(-10/omega_buckets_divisor):1:round(10/omega_buckets_divisor) # omega = [-10, 10] with 501 values evenly spaced
    for x=0:1:round(6.28/theta_buckets_divisor) # theta = [0, 2pi] with 501 values evenly spaced
        state = (x,y)
        action = get(UtilityMap,state,(-1,-1,0))[2]
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
action_found_rate = (NUM_STATES_OMEGA*NUM_STATES_THETA-num_no_action_found)/(NUM_STATES_OMEGA*NUM_STATES_THETA)
println("Rate of best action found: ", action_found_rate*100, "%")
