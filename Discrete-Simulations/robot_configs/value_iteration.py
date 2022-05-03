#rewards_per_tile = {-5:1, -3: 1, -2: -1, -1: -1, 0: 3, 1: 6, 2: 8, 3: -1}
rewards_per_tile = {-2: -1, -1: -1, 0: 3, 1: 6, 2: 8, 3: -1}
# error tolerance
theta = 0.2
# discount factor
gamma = 0.9


def get_states(grid):
    # empty dict, where key=(x,y) coordinates and value='type of block'
    states = {}
    # begin iterating the grid to create states dict
    row_id = 0
    for row in grid:
        row_id += 1
        col_id = 0
        for block_type in row:
            col_id += 1
            states[(row_id, col_id)] = block_type

    return states


def get_next_state_dicts(states, initial_position):
    # create dict, where key='x,y coordinates' (state),
    # and value=tuple of 'type of block', 'action' (states[state], (+-x, +-y))
    state_action_tuples = {}

    # for each state check if it can be reached from the current position
    # if reachable -> add it to state_action_tuples
    for state in states:
        if (state[0]-1, state[1]) == initial_position:
            state_action_tuples[state] = (states[state], (+1, 0))
        elif (state[0]+1, state[1]) == initial_position:
            state_action_tuples[state] = (states[state], (-1, 0))
        elif (state[0], state[1]-1) == initial_position:
            state_action_tuples[state] = (states[state], (0, +1))
        elif (state[0], state[1]+1) == initial_position:
            state_action_tuples[state] = (states[state], (0, -1))

    return state_action_tuples


def value_iteration(states, policy, V_values):
    while True:
        delta = 0
        # loop through states s
        for state in states:
            # save previous V_value
            v = V_values[state]
            # get next states and their associated type and action to reach it
            next_states_dict = get_next_state_dicts(states, state)
            # if no next states s^prime skip current state s
            if len(next_states_dict) <= 0:
                continue

            # start comuting the new V_value for the current state
            # to circumvent the problem of types with value <-2 set their type to 1
            # (I observed that states with those values are the ones that the robot is currently on => have been dirty)
            if states[state] < -2:
                V_values[state] = 1
            else:
                V_values[state] = rewards_per_tile[states[state]]
            for next_state in next_states_dict:
                V_values[state] += gamma * V_values[next_state]

            # print(v)
            # print(V_values[state])
            # print('----------------')

            # update delta
            delta = max(delta, abs(v - V_values[state]))

        if delta < theta:
            break

    # problem: V_values become inf -> suggests that
    # abs(v - V_values[state]) < theta when both v and V_values[state] become inf
    # currently small values for theta lead to inf
    print(V_values)

    # start updating policy
    for state in states:
        # get next states and their associated type and action to reach it
        next_states_dict = get_next_state_dicts(states, state)
        # if no next states s^prime skip current state s
        if len(next_states_dict) <= 0:
            continue
        # dictionary to store the value for each action
        actions = {}

        # loop through all next states
        for next_state in next_states_dict:
            # get the action leading to that state
            action = next_states_dict[next_state][1]
            # to circumvent the problem of types with value <-2 set their type to 1
            if states[state] < -2:
                actions[action] = 1 + gamma * V_values[next_state]
            else:
                actions[action] = rewards_per_tile[states[state]] +\
                                                           gamma * V_values[next_state]

        # update the policy for that state with the key associated with the largest value from dict actions
        policy[state] = max(actions, key=actions.get)

    return policy


def robot_epoch(robot):
    # get all states and their rewards in dictionary states
    states = get_states(robot.grid.cells)
    # initialize policy for every state
    policy = {state: None for state in states}
    # initialize V function values
    V_values = {state: 0 for state in states}
    # start value iteration
    policy = value_iteration(states, policy, V_values)
    # get the move (up or down or left or right) for the policy for the current position
    move = policy[robot.pos]

    print('Policy:'+str(move))
    # Find out how we should orient ourselves:
    new_orient = list(robot.dirs.keys())[list(robot.dirs.values()).index(move)]
    # Orient ourselves towards the dirty tile:
    while new_orient != robot.orientation:
        # If we don't have the wanted orientation, rotate clockwise until we do:
        # print('Rotating right once.')
        robot.rotate('r')
    # Move:
    robot.move()
