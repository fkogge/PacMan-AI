# multiAgents.py
# --------------
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

import random, util

from game import Agent
from util import manhattanDistance

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        foodReward = 50
        ghostIncentive = 25
        capsuleIncentive = 1000
        rewardMultiplier = 3

        score = successorGameState.getScore()

        for foodPos in newFood.asList():
            distance = manhattanDistance(newPos, foodPos)
            if distance == 0:
                score += foodReward
            else:
                score += (1 / distance)

        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
            ghostPos = ghostState.getPosition()
            distance = manhattanDistance(newPos, ghostPos)
            if distance <= 1:
                if scaredTime > 0:
                    score += ghostIncentive
                else:
                    score -= ghostIncentive

        for capsulePos in successorGameState.getCapsules():
            distance = manhattanDistance(newPos, capsulePos)
            if distance == 0:
                score += capsuleIncentive
            else:
                score += ((1 / distance) * rewardMultiplier)

        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    def isTerminal(self, state, depth):
        """
        Returns whether this state is terminal (in game tree, this would be
        a leaf node with a static value).
        :param state: current game state
        :param depth: depth of the game tree
        :return: true if terminal node, otherwise false
        """
        return state.isWin() or state.isLose() or depth == self.depth

    def getNextAgentIndex(self, agentIndex, numAgents):
        """
        Get the next agent index - PacMan (0) if we looked at all ghost states,
        otherwise it increments to the next ghost index
        :param agentIndex: current agent index
        :param numAgents: number of agents in the game
        :return: next agent index
        """
        return self.index if agentIndex + 1 == numAgents else agentIndex + 1

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        self.numAgents = gameState.getNumAgents()
        maxValue, maxAction = float('-inf'), float('-inf')

        # Calculate minimax value on each action for PacMan
        for action in gameState.getLegalActions(self.index):
            successorState = gameState.generateSuccessor(self.index, action)
            value = self.miniMaxValue(successorState, 0, self.index)
            if value > maxValue:
                maxValue, maxAction = value, action

        return maxAction

    def miniMaxValue(self, nextGameState, depth, agentIndex):
        """
        Calculates the minimax value for the next game state
        :param nextGameState: next game state
        :param depth: current depth of the minimax tree
        :param agentIndex: current agent index
        :return: minimax value
        """
        nextAgentIndex = self.getNextAgentIndex(agentIndex, self.numAgents)

        if nextAgentIndex == self.index:
            # PacMan -> maximize action and increment depth since we've
            # completed one search ply
            return self.maxValue(nextGameState, depth + 1, nextAgentIndex)
        else:
            # Ghost -> minimize action and remain in the same search ply
            return self.minValue(nextGameState, depth, nextAgentIndex)

    def minValue(self, gameState, depth, agentIndex):
        """
        Calculates the minimum value for this game state (minimizer node in the
        minimax tree)
        :param gameState: current game state
        :param depth: current depth of the minimax tree
        :param agentIndex: current agent index
        :return: minimum value
        """
        if self.isTerminal(gameState, depth):
            return self.evaluationFunction(gameState)

        value = float('inf')
        for action in gameState.getLegalActions(agentIndex):
            successorState = gameState.generateSuccessor(agentIndex, action)
            value = min(value, self.miniMaxValue(successorState, depth, agentIndex))
        return value

    def maxValue(self, gameState, depth, agentIndex):
        """
        Calculates the maximum value for this game state (maximizer node in the
        minimax tree)
        :param gameState: current game state
        :param depth: current depth of the minimax tree
        :param agentIndex: current agent index
        :return: maximum value
        """
        if self.isTerminal(gameState, depth):
            return self.evaluationFunction(gameState)

        value = float('-inf')
        for action in gameState.getLegalActions(agentIndex):
            successorState = gameState.generateSuccessor(agentIndex, action)
            value = max(value, self.miniMaxValue(successorState, depth, agentIndex))
        return value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        self.numAgents = gameState.getNumAgents()
        maxValue, maxAction = float('-inf'), float('-inf')

        # Calculate minimax value on each action for PacMan
        for action in gameState.getLegalActions(self.index):
            successorState = gameState.generateSuccessor(self.index, action)
            value = self.miniMaxValue(
                successorState, 0, self.index, alpha=maxValue, beta=float('inf')
            )
            if value > maxValue:
                maxValue, maxAction = value, action

        return maxAction

    def miniMaxValue(self, nextGameState, depth, agentIndex, alpha, beta):
        """
        Calculates the minimax value for the next game state
        :param nextGameState: next game state
        :param depth: current depth of the minimax tree
        :param agentIndex: current agent index
        :param alpha: maximizer node's best option on path to root
        :param beta: minimizer node's best option on path to root
        :return: minimax value
        """
        nextAgentIndex = self.getNextAgentIndex(agentIndex, self.numAgents)

        if nextAgentIndex == self.index:
            # PacMan -> maximize action and increment depth since we've
            # completed one search ply
            return self.maxValue(nextGameState, depth + 1, nextAgentIndex, alpha, beta)
        else:
            # Ghost -> minimize action and remain in the same search ply
            return self.minValue(nextGameState, depth, nextAgentIndex, alpha, beta)

    def minValue(self, gameState, depth, agentIndex, alpha, beta):
        """
        Calculates the minimum value for this game state (minimizer node in the
        minimax tree)
        :param gameState: current game state
        :param depth: current depth of the minimax tree
        :param agentIndex: current agent index
        :param alpha: maximizer node's best option on path to root
        :param beta: minimizer node's best option on path to root
        :return: minimum value
        """
        if self.isTerminal(gameState, depth):
            return self.evaluationFunction(gameState)

        value = float('inf')
        for action in gameState.getLegalActions(agentIndex):
            successorState = gameState.generateSuccessor(agentIndex, action)
            value = min(value, self.miniMaxValue(successorState, depth, agentIndex, alpha, beta))

            # Max-node will pick alpha anyways, so prune
            if value < alpha:
                return value

            beta = min(beta, value)

        return value

    def maxValue(self, gameState, depth, agentIndex, alpha, beta):
        """
        Calculates the maximum value for this game state (maximizer node in the
        minimax tree)
        :param gameState: current game state
        :param depth: current depth of the minimax tree
        :param agentIndex: current agent index
        :param alpha: maximizer node's best option on path to root
        :param beta: minimizer node's best option on path to root
        :return: maximum value
        """
        if self.isTerminal(gameState, depth):
            return self.evaluationFunction(gameState)

        value = float('-inf')
        for action in gameState.getLegalActions(agentIndex):
            successorState = gameState.generateSuccessor(agentIndex, action)
            value = max(value, self.miniMaxValue(successorState, depth, agentIndex, alpha, beta))

            # Min-node will pick beta anyways, so prune
            if value > beta:
                return value

            alpha = max(alpha, value)

        return value


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        self.numAgents = gameState.getNumAgents()
        maxValue, maxAction = float('-inf'), float('-inf')

        for action in gameState.getLegalActions(self.index):
            successorState = gameState.generateSuccessor(self.index, action)
            value = self.expectiMaxValue(successorState, 0, self.index)
            if value > maxValue:
                maxValue, maxAction = value, action
        return maxAction

    def expectiMaxValue(self, nextGameState, depth, agentIndex):
        """
        Calculates the expectiMax value for the next game state
        :param nextGameState: next game state
        :param depth: current depth of the expectiMax tree
        :param agentIndex: current agent index
        :return: expectiMax value
        """
        nextAgentIndex = self.getNextAgentIndex(agentIndex, self.numAgents)

        if nextAgentIndex == self.index:
            # PacMan -> maximize action and increment depth since we've
            # completed one search ply
            return self.maxValue(nextGameState, depth + 1, nextAgentIndex)
        else:
            # Ghost -> get expected value and remain in the same search ply
            return self.expectedValue(nextGameState, depth, nextAgentIndex)

    def expectedValue(self, gameState, depth, agentIndex):
        """
        Calculates the expected value for this game state (chance node in the
        minimax tree), using uniform probability distribution.
        :param gameState: current game state
        :param depth: current depth of the minimax tree
        :param agentIndex: current agent index
        :return: expected value
        """
        if self.isTerminal(gameState, depth):
            return self.evaluationFunction(gameState)

        value = 0
        legalActions = gameState.getLegalActions(agentIndex)
        # Uniform probability of choosing any of the given legal actions
        probability = 1 / len(legalActions)

        for action in legalActions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            value += probability * self.expectiMaxValue(successorState, depth, agentIndex)

        return value

    def maxValue(self, gameState, depth, agentIndex):
        """
        Calculates the maximum value for this game state (maximizer node in the
        minimax tree)
        :param gameState: current game state
        :param depth: current depth of the minimax tree
        :param agentIndex: current agent index
        :return: maximum value
        """
        if self.isTerminal(gameState, depth):
            return self.evaluationFunction(gameState)

        value = float('-inf')
        for action in gameState.getLegalActions(agentIndex):
            successorState = gameState.generateSuccessor(agentIndex, action)
            value = max(value, self.expectiMaxValue(successorState, depth, agentIndex))

        return value


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Calculates the score by evaluating the amount of food left,
    number of capsules left, the distance to each food, the distance to each
    capsule, and the scared times of the ghosts. Multipliers are used for
    food and capsule rewards.
    """
    # Extract useful info from current game state
    pos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    capsules = currentGameState.getCapsules()

    foodReward = 50
    scaredTimeIncentive = 50
    foodMulitplier = 5
    capsuleMultiplier = 10

    # Initialize to current score
    score = currentGameState.getScore()

    # The more food that is left, the less the score increases
    if foodList:
        score += 1 / (len(foodList) * foodMulitplier)

    # The more capsules that are left, the less the score increases
    if capsules:
        score += 1 / (len(capsules) * capsuleMultiplier)

    # Consider distance to each food
    for foodPos in foodList:
        distance = manhattanDistance(pos, foodPos)
        if distance == 0:
            score += foodReward
        else:
            # Less score is added if distance to this food is greater
            score += 1 / (distance * foodMulitplier)

    # Consider how long ghosts are scared
    for scaredTime in scaredTimes:
        if scaredTime > 0:
            score += scaredTimeIncentive

    return score

# Abbreviation
better = betterEvaluationFunction
