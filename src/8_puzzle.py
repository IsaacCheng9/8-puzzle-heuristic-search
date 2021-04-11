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


# Uses the start and goal states from figure 1 of the worksheet.
def main():
    heuristic = choose_heuristic()
    start = np.array([7, 2, 4, 5, 0, 6, 8, 3, 1])
    goal = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    print(heuristic, "Distance heuristic chosen.\nStart State:", start,
          "\nGoal State:", goal)

    # Calculate Hamming Distance heuristic.
    if heuristic == "Hamming":
        optimal = np.sum(start != goal) - 1
        print(optimal)

    moves = np.array([("up", [0, 1, 2], -3),
                      ("down", [6, 7, 8], 3),
                      ("left", [0, 3, 6], -1),
                      ("right", [2, 5, 8], 1)],
                     dtype=[("move", str, 1),
                            ("position", list),
                            ("head", int)])
    dt_state = [("board", list),
                ("parent", int),
                ("gn", int),
                ("hn", int)]


if __name__ == "__main__":
    main()
