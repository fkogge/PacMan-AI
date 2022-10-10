# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    # LIFO stack to achieve DFS
    return uninformedGraphSearch(problem, util.Stack())

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    # FIFO queue to achieve BFS
    return uninformedGraphSearch(problem, util.Queue())

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    # Priority queue to achieve UCS
    return uninformedGraphSearch(problem, util.PriorityQueue(), isUcs=True)

def uninformedGraphSearch(problem, fringe, isUcs=False):
    """
    Helper function for uninformed graph search. Change the search algorithm
    (DFS or BFS) by passing in a different data structure for the fringe.

    :param problem: game problem type
    :param fringe: data structure to store the unexpanded nodes
    :param isUcs: true if we are doing a uniform-cost search
    :return: list of actions to get to the goal state
    """
    """

    """
    startState = problem.getStartState()
    # check if we start at the goal state
    if problem.isGoalState(startState):
        return []

    visited = set()  # keep track of visited locations
    if isUcs:
        fringe.push((startState, []), 0)  # add initial cost for UCS
    else:
        fringe.push((startState, []))  # omit cost for general DFS or BFS

    while not fringe.isEmpty():
        currentState, actionsTaken = fringe.pop()

        # if pacman found the food pellet, return actions taken to get there
        if problem.isGoalState(currentState):
            return actionsTaken

        if currentState not in visited:
            visited.add(currentState)  # mark location as visited
            successorList = problem.getSuccessors(currentState)

            for successorState, successorAction, _ in successorList:
                # record path to successor by taking actions taken so far
                # plus the action required to get to the successor from current
                actionsToSuccessor = actionsTaken + [successorAction]
                successorNode = (successorState, actionsToSuccessor)
                if isUcs:
                    fringe.push(successorNode,
                                problem.getCostOfActions(actionsToSuccessor))
                else:
                    fringe.push(successorNode)

    # no path to goal node if we reach here (failure)
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    if problem.isGoalState(problem.getStartState()):
        return []

    visited = set()  # to keep track of visited locations
    fringe = util.PriorityQueue()  # priority queue to perform UCS
    # initialize priority queue with start state and initial cost = 0
    fringe.push((problem.getStartState(), []), 0)

    while not fringe.isEmpty():
        currentState, actionsTaken = fringe.pop()
        # if pacman found the food pellet, return actions taken to get there
        if problem.isGoalState(currentState):
            return actionsTaken

        if currentState not in visited:
            # mark location as visited
            visited.add(currentState)
            successorList = problem.getSuccessors(currentState)

            for successorState, successorAction, _ in successorList:
                # actions taken so far plus the action required to get to the
                # successor from current
                actionsToSuccessor = actionsTaken + [successorAction]
                costToSuccessor = problem.getCostOfActions(actionsToSuccessor)  # g(n)
                heuristicVal = heuristic(successorState, problem)  # h(n)
                totalCost = costToSuccessor + heuristicVal  # f(n) = g(n) + h(n)
                fringe.push(
                    (successorState, actionsToSuccessor),
                    totalCost
                )

    # no path to goal node if we reach here (failure)
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
