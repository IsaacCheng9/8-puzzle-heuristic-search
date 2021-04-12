from copy import deepcopy

import numpy as np


def choose_heuristic():
    """
    Gets the user to select either the Manhattan or Hamming Distance heuristic.

    Returns:
        The type of heuristic to use for solving the 8-puzzle problem.
    """
    heuristic_input = ""
    while heuristic_input not in ("m", "h"):
        heuristic_input = input(
            "\nThe Manhattan Distance heuristic is based on number of squares "
            "between itself and the goal position, whereas the Hamming "
            "Distance heuristic is based on total number of misplaced tiles.\n"
            "Would you like to use the Manhattan Distance heuristic (m) or "
            "the Hamming Distance heuristic (h)?\n").lower()
        if heuristic_input not in ("m", "h"):
            print("Invalid input! Please enter 'm' for the Manhattan Distance "
                  "heuristic, or 'h' for the Hamming Distance heuristic.")
        elif heuristic_input == "m":
            return "Manhattan"
        else:
            return "Hamming"


def assign_coordinates(board):
    """
    Assigns coordinates to each digit to calculate the Manhattan Distance.

    Args:
        board: The state of the board.

    Returns:
        Coordinates to calculate the Manhattan Distance.
    """
    coordinate = np.array(range(9))
    for x, y in enumerate(board):
        coordinate[y] = x
    return coordinate


def calculate_heuristic(heuristic, current, goal) -> int:
    if heuristic == "Manhattan":
        result = manhattan_heuristic(current, goal)
    else:
        result = hamming_heuristic(current, goal)
    return result


def manhattan_heuristic(current, goal) -> int:
    manhattan = abs(current // 3 - goal // 3) + abs(current % 3 - goal % 3)
    return sum(manhattan[1:])


def hamming_heuristic(current, goal) -> int:
    return np.sum(current != goal) - 1


def generate_steps(state):
    num_steps = np.array([], int).reshape(-1, 9)
    last = len(state) - 1
    while last != -1:
        num_steps = np.insert(num_steps, 0, state[last]["board"], 0)
        last = int(state[last]["parent"])
    return num_steps.reshape(-1, 3, 3)


def search(heuristic, start, goal):
    moves = np.array([("up", [0, 1, 2], -3),
                      ("down", [6, 7, 8], 3),
                      ("left", [0, 3, 6], -1),
                      ("right", [2, 5, 8], 1)],
                     dtype=[("move", str, 1),
                            ("position", list),
                            ("delta", int)])
    dt_state = [("board", list),
                ("parent", int),
                ("gn", int),
                ("hn", int)]

    start_c = assign_coordinates(start)
    goal_c = assign_coordinates(goal)
    print(start_c)
    print(goal_c)
    parent = -1
    g_function = 0
    h_function = calculate_heuristic(heuristic, start_c, goal_c)
    state = np.array([(start, parent, g_function, h_function)],
                     dt_state)

    dt_priority = [("position", int),
                   ("f_function", int)]
    priority = np.array([(0, h_function)], dt_priority)

    while True:
        # Sorts the priority queue.
        priority = np.sort(priority, kind="mergesort",
                           order=["f_function", "position"])
        # Explores the first node from the priority queue, and removes it from
        # the priority queue.
        position, f_function = priority[0]
        priority = np.delete(priority, 0, 0)
        board, parent, g_function, h_function = state[position]
        board = np.array(board)
        location = int(np.where(board == 0)[0])
        g_function += 1

        for m in moves:
            if location not in m["position"]:
                # Copies the current state of the board.
                new_state = deepcopy(board)
                # Performs the move.
                new_state[location], new_state[location + m["delta"]] = \
                    new_state[location + m["delta"]], new_state[location]
                if not (np.all(list(state["board"]) == new_state, 1)).any():
                    h_function = calculate_heuristic(
                        heuristic, assign_coordinates(new_state), goal_c)
                    # Generates and adds the new state to the queue.
                    queue = np.array(
                        [(new_state, position, g_function, h_function)],
                        dt_state)
                    state = np.append(state, queue, 0)
                    f_function = g_function + h_function
                    # Adds the new state to the priority queue.
                    queue = np.array([(len(state) - 1, f_function)],
                                     dt_priority)
                    priority = np.append(priority, queue, 0)

                    # Stops the search if the goal state has been achieved.
                    if np.array_equal(new_state, goal):
                        print("Goal state has been achieved!")
                        return state, len(priority)


def main():
    """
    Runs the A* algorithm to solve the 8-puzzle problem.
    """
    heuristic = choose_heuristic()
    start = np.array([7, 2, 4, 5, 0, 6, 8, 3, 1])
    goal = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    print(heuristic, "Distance heuristic chosen.\nStart State:", start,
          "\nGoal State:", goal)

    state, explored = search(heuristic, start, goal)
    optimal = generate_steps(state)
    print("Total States Generated:", len(state), "\nTotal States Explored:",
          len(state) - explored, "\nTotal Steps for Optimal Solution:",
          len(optimal) - 1)


if __name__ == "__main__":
    main()
