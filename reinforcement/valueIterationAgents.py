# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        """Runs the value iteration using batch style update to values."""
        for k in range(self.iterations):
            nextValues = util.Counter()
            for state in self.mdp.getStates():
                maxAction = self.getAction(state)
                if maxAction:
                    nextValues[state] = self.getQValue(state, maxAction)

            self.values = nextValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        qValue = 0

        for nextState, transProb in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, nextState)
            qValue += transProb * (reward + self.discount * self.values[nextState])

        return qValue


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):
            return None

        possibleActions = self.mdp.getPossibleActions(state)
        bestAction, maxValue = float('-inf'), float('-inf')

        for action in possibleActions:
            qValue = self.getQValue(state, action)
            if qValue > maxValue:
                bestAction, maxValue = action, qValue

        return bestAction



    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        """Runs asynchronous value iteration"""
        states = self.mdp.getStates()

        for k in range(self.iterations):
            # Wrap back around to first state
            state = states[k % len(states)]
            maxAction = self.getPolicy(state)
            if maxAction:  # Make sure state is not terminal
                self.values[state] = self.getQValue(state, maxAction)


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        """
        Runs the value iteration. Implemented using the provided pseudocode
        for prioritized sweeping value iteration.
        """
        states = self.mdp.getStates()
        predecessors = self.computePredecessors(states)
        queue = util.PriorityQueue()

        # Calculate absolute difference between current value of state
        # and highest Q-value across, then add state to priority queue
        for state in states:
            if not self.mdp.isTerminal(state):
                maxQValue = self.getMaxQValue(state)
                diff = abs(self.values[state] - maxQValue)
                queue.update(state, priority=-diff)

        for iteration in range(self.iterations):
            if queue.isEmpty():
                return

            state = queue.pop()

            # Update value for the state
            if not self.mdp.isTerminal(state):
                self.values[state] = self.getMaxQValue(state)

            # Calculate absolute difference between current value of predecessor
            # state and highest Q-value, then add predessor state to priority queue
            for predecessorSet in predecessors.values():
                for pred in predecessorSet:
                    maxQValue = self.getMaxQValue(pred)
                    diff = abs(self.values[pred] - maxQValue)
                    if diff > self.theta:
                        queue.update(pred, priority=-diff)


    def getMaxQValue(self, state):
        """
        Compute the highest Q-value across all possible actions from the state
        :param state: state to evaluate
        :return: highest Q-value
        """
        maxQValue = float('-inf')
        for action in self.mdp.getPossibleActions(state):
            maxQValue = max(maxQValue, self.getQValue(state, action))
        return maxQValue

    def computePredecessors(self, states):
        """
        Computes the predecessor states for each of the states.

        From spec: "we define the predecessors of a state s as all states that
        have a nonzero probability of reaching s by taking some action a"

        :param states: list of states to compute predecessors of
        :return: dictionary of states (tuple) mapped to their predecessor states (set)
        """
        predecessors = {}  # Key: state, Value: set of predecessor states

        for state in states:
            for action in self.mdp.getPossibleActions(state):
                for nextState, transProb in self.mdp.getTransitionStatesAndProbs(state, action):
                    if transProb > 0:
                        if nextState not in predecessors:
                            predecessors[nextState] = set()
                        predecessors[nextState].add(state)

        return predecessors