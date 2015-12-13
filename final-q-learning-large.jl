# This uses q-learning to create the best policy for the large data set
# 1) Use the given data to 'explore' the different states in the world
# 2) At each state, update the q-value based on that state's observed rewards and expected rewards from future states.
# 3) Extract the best policy (action that maximizes utility at each state)
# Note: Only explores the world based on the (s,a,r,s') given in the data, no exploration done on its own. For further improvement, should explore the world by randomizing actions through the world instead of just following the steps given in the data set.

using DataFrames
using DataStructures

#TUNING PARAMETERS
discountFactor = 1.0
numIterations = 10
learningRate = 0.9
println("Number of Iterations ", numIterations)


#Defining Data Set
DATA_SET = "large"
outputFile = open(string(DATA_SET,".policy"),"w")
NUM_STATES = 531441
NUM_ACTIONS = 157
action_array = collect(1:1:NUM_ACTIONS)

#Setup
data = readtable(string(DATA_SET,".csv"));
q_values = zeros(NUM_STATES, NUM_ACTIONS) # matrix with utility values for each (s,a) pair
possible_actions_map = Dict(); # all possible actions from each state, (state) --> [list of valid actions]

#"Explore" world by iterating over data
println("Exploring world!")

for i = 1:1:numIterations
    println("** Iteration # ", i)
    for i = 1:1:size(data,1)
        state = data[i,1]
        state_p = data[i,3]
        action = data[i,2]
        reward = data[i,4]
        #append new action to state in possible_actions_map
        if(haskey(possible_actions_map, state))
            possible_actions_map[state] = append!(possible_actions_map[state], [action]) 
        else
            possible_actions_map[state] = [action]
        end
        s_a = (state, action)

        row_of_action_states = q_values[state_p,:] #for next state
        best_action = indmax(row_of_action_states)

        max_q_next_state = q_values[state,action]
        best_sp_ap = (state_p, best_action)

        #println("Best next state-action pair ", best_sp_ap, " with q value: ", max_q_next_state)

        q_values[state,action] = q_values[state,action] + learningRate*(reward + discountFactor*max_q_next_state - q_values[state,action])
    end
end

#Extract best policy

println("Extracting best policy! To output file ", outputFile)

BestPolicy = Dict();
start_of_file = true;
num_no_action_found = 0;
for x=1:1:NUM_STATES
    state = x
    row_of_action_states = q_values[state,:]
    best_action = indmax(row_of_action_states)
    if q_values[state,best_action] == 0
        num_no_action_found += 1
        best_action = rand(action_array)  #pick random action
    end
    if !start_of_file
        write(outputFile, string("\n", best_action))
    else
        write(outputFile, string(best_action))  # makes sure that we have no trailing empty lines in output
        start_of_file = false
    end
end

close(outputFile)

println("---- Printing Results ----")
println("Num States with no best action policy found: ", num_no_action_found)
action_found_rate = (NUM_STATES-num_no_action_found)/NUM_STATES
println("Rate of best action found: ", action_found_rate*100, "%")
