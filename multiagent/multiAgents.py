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


from util import manhattanDistance
from game import Directions
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
        #print('successorGameState: ' + str(type(successorGameState)))
        # print('newPos: ' + str(newPos))
        # print('newFood: ' + str(newFood.asList()))
        # print('newGhostStates: ' + str(len(newGhostStates)))
        #print('newScaredTimes: ' + str(newScaredTimes))
        "*** YOUR CODE HERE ***"

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


        #return successorGameState.getScore()
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
        return state.isWin() or state.isLose() or depth == self.depth

    def getNextAgentIndex(self, agentIndex, numAgents):
        # Set next agent to PacMan if we looked at all ghost states, else
        # just increment to next ghost index
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

        for action in gameState.getLegalActions(self.index):
            successorState = gameState.generateSuccessor(self.index, action)
            value = self.miniMaxValue(successorState, self.index, 1)
            if value > maxValue:
                maxValue, maxAction = value, action
        return maxAction

    def miniMaxValue(self, gameState, depth, nextAgentIndex):
        if nextAgentIndex == self.index:
            # Next agent is PacMan -> maximize action and increment depth
            # since we've completed one search ply
            return self.maxValue(gameState, depth + 1)
        else:
            # Next agent is ghost -> so minimize action and we remain in
            # the same search ply
            return self.minValue(gameState, depth, nextAgentIndex)

    def minValue(self, gameState, depth, agentIndex):
        if self.isTerminal(gameState, depth):
            return self.evaluationFunction(gameState)

        value = float('inf')
        for action in gameState.getLegalActions(agentIndex):
            successorState = gameState.generateSuccessor(agentIndex, action)
            nextAgentIndex = self.getNextAgentIndex(agentIndex, self.numAgents)
            value = min(
                value,
                self.miniMaxValue(successorState, depth, nextAgentIndex)
            )
        return value

    def maxValue(self, gameState, depth):
        if self.isTerminal(gameState, depth):
            return self.evaluationFunction(gameState)

        value = float('-inf')
        for action in gameState.getLegalActions(self.index):
            successorState = gameState.generateSuccessor(self.index, action)
            value = max(
                value,
                self.miniMaxValue(successorState, depth, 1)
            )
        return value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        self.numAgents = gameState.getNumAgents()
        maxValue, maxAction = float('-inf'), float('-inf')

        for action in gameState.getLegalActions(self.index):
            successorState = gameState.generateSuccessor(self.index, action)
            value = self.miniMaxValue(
                successorState, self.index, 1, alpha=maxValue, beta=float('inf')
            )
            if value > maxValue:
                maxValue, maxAction = value, action

        return maxAction

    def miniMaxValue(self, gameState, depth, nextAgentIndex, alpha, beta):
        if nextAgentIndex == self.index:
            # If next agent is PacMan, maximize action and increment depth
            # since we've completed one search ply
            return self.maxValue(gameState, depth + 1, alpha, beta)
        else:
            # Else, next agent is ghost, so minimize action and we remain in
            # the same search ply
            return self.minValue(gameState, depth, nextAgentIndex, alpha, beta)

    def minValue(self, gameState, depth, agentIndex, alpha, beta):
        if self.isTerminal(gameState, depth):
            return self.evaluationFunction(gameState)

        value = float('inf')
        for action in gameState.getLegalActions(agentIndex):
            successorState = gameState.generateSuccessor(agentIndex, action)
            nextAgentIndex = self.getNextAgentIndex(agentIndex, self.numAgents)
            value = min(
                value,
                self.miniMaxValue(successorState, depth, nextAgentIndex, alpha, beta)
            )

            # Max-node will pick alpha anyways, so prune
            if value < alpha:
                return value

            beta = min(beta, value)

        return value

    def maxValue(self, gameState, depth, alpha, beta):
        if self.isTerminal(gameState, depth):
            return self.evaluationFunction(gameState)

        value = float('-inf')
        for action in gameState.getLegalActions(self.index):
            successorState = gameState.generateSuccessor(self.index, action)
            value = max(
                value,
                self.miniMaxValue(successorState, depth, 1, alpha, beta)
            )

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
            value = self.expectiMaxValue(successorState, self.index, 1)
            if value > maxValue:
                maxValue, maxAction = value, action
        return maxAction

    def expectiMaxValue(self, gameState, depth, nextAgentIndex):
        if nextAgentIndex == self.index:
            # If next agent is PacMan, maximize action and increment depth
            # since we've completed one search ply
            return self.maxValue(gameState, depth + 1)
        else:
            # Else, next agent is ghost, so calculate expected value and we
            # remain in the same search ply
            return self.expectedValue(gameState, depth, nextAgentIndex)

    def expectedValue(self, gameState, depth, agentIndex):
        if self.isTerminal(gameState, depth):
            return self.evaluationFunction(gameState)

        value = 0
        legalActions = gameState.getLegalActions(agentIndex)
        # Uniform probability of choosing any of the given legal actions
        probability = 1 / len(legalActions)

        for action in legalActions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            nextAgentIndex = self.getNextAgentIndex(agentIndex, self.numAgents)
            value += probability * self.expectiMaxValue(successorState, depth, nextAgentIndex)

        return value

    def maxValue(self, gameState, depth):
        if self.isTerminal(gameState, depth):
            return self.evaluationFunction(gameState)

        value = float('-inf')
        for action in gameState.getLegalActions(self.index):
            successorState = gameState.generateSuccessor(self.index, action)
            value = max(
                value,
                self.expectiMaxValue(successorState, depth, 1)
            )

        return value


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
