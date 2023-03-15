import random
import numpy as np


# UCB1 action selection algorithm
def ucb_action(mcts, current_node, greedy, filterUnder = 0):
    best_actions = []
    best_q_value = -np.inf
    mapping = current_node.action_map

    N = mapping.total_visit_count
    log_n = np.log(N + 1)

    actions = list(mapping.entries.values())
    random.shuffle(actions)
    for action_entry in actions:

        # Skip illegal actions
        if not action_entry.is_legal or action_entry.bin_number < filterUnder:
            continue

        current_q = action_entry.mean_q_value

        # If the UCB coefficient is 0, this is greedy Q selection
        if not greedy:
            current_q += mcts.find_fast_ucb(N, action_entry.visit_count, log_n)

        if current_q >= best_q_value:
            if current_q > best_q_value:
                best_actions = []
            best_q_value = current_q
            # best actions is a list of Discrete Actions
            best_actions.append(action_entry.get_action())
    assert best_actions.__len__() is not 0
    
    # Break ties with mean:
    # if best_actions.__len__() > 1:
    #     newActionList = []
    #     best_mean_q = -np.inf
    #     for action in best_actions:
    #         current_q = action_entry.mean_q_value
    #         if current_q >= best_mean_q:
    #             if current_q > best_mean_q:
    #                 best_actions = []
    #                 best_mean_q = current_q
    #             newActionList.append(action_entry.get_action())
    #     best_actions = newActionList
    # at each iteration print out 16 action (mean q values)
    if greedy:
        print(best_q_value)
    
    return random.choice(best_actions)


def e_greedy(current_node, epsilon):
    best_actions = []
    best_q_value = -np.inf
    mapping = current_node.action_map

    actions = list(mapping.entries.values())
    random.shuffle(actions)

    if np.random.uniform(0, 1) < epsilon:
        for action_entry in actions:
            if not action_entry.is_legal:
                continue
            else:
                return action_entry.get_action()
        # No legal actions
        raise RuntimeError('No legal actions to take')
    else:
        # Greedy choice
        for action_entry in actions:
            # Skip illegal actions
            if not action_entry.is_legal:
                continue

            current_q = action_entry.mean_q_value

            if current_q >= best_q_value:
                if current_q > best_q_value:
                    best_actions = []
                best_q_value = current_q
                # best actions is a list of Discrete Actions
                best_actions.append(action_entry.get_action())

        assert best_actions.__len__() is not 0

        return random.choice(best_actions)
