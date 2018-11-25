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
import math
from game import Agent

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

        "*** YOUR CODE HERE ***"
        #check the closest ghost distance
        ghostNearMe = min([manhattanDistance(s.getPosition(),newPos) for s in newGhostStates])
        #defeat is imminent
        if ghostNearMe <= 1:
          return -1000

        foodNearMe=0
        foodDistanceList = [manhattanDistance(newPos,f) for f in newFood.asList()]
        if not foodDistanceList:
          #won the game
          foodNearMe = 0.0
        else:
          foodNearMe = min(foodDistanceList)
        weight = successorGameState.getScore() - foodNearMe - 100*(successorGameState.getNumFood() - currentGameState.getNumFood())
        if weight == 0:
          return 1
        return weight


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
        """
        "*** YOUR CODE HERE ***"
        mm = self.minmaxFunction(0,gameState,0)
        return mm[-1]
        
    def minmaxFunction(self,agent, gameState, depth):
      #check the terminal case, when no further recursion is required
      if depth == self.depth:
        ans = list()
        ans.append(self.evaluationFunction(gameState))
        return ans
      
      #keep all the possible states in a list
      possibleStates = list()
      for move in gameState.getLegalActions(agent):
        temp = list()
        temp.append(gameState.generateSuccessor(agent, move))
        temp.append(move)
        possibleStates.append(temp)

      if len(possibleStates) == 0:
        possibleStates = [(gameState, None)]
      
      #if pacman's turn , means agent is 0, we choose the max of all possible
      if agent == 0:
        maxVal = [-100000000]
        
        for gState,action in possibleStates:
          nextVal = self.minmaxFunction((agent+1)%gameState.getNumAgents(), gState, depth)
          nextVal.append(action)
          maxVal = max(maxVal,nextVal)
        return (maxVal)
      #else we choose the min value
      else:
        minValue = [100000000000]
        
        for gState,action in possibleStates:
          nextVal = self.minmaxFunction((agent+1)%gameState.getNumAgents(), gState, depth + ((agent+1)%gameState.getNumAgents()==0))
          nextVal.append(action)
          minValue = min(minValue,nextVal)
        return (minValue)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        ab  = self.alphaBetaFunction(0,gameState,0,-100000000000.0,100000000000.0)
        return ab[-1]

    def alphaBetaFunction(self,agent, gameState, depth, alpha, beta):
      if depth == self.depth :
        ans = list()
        ans.append(self.evaluationFunction(gameState))
        return ans
      
      if gameState.isWin() or gameState.isLose():
        ans = list()
        ans.append(self.evaluationFunction(gameState))
        return ans

      actions = gameState.getLegalActions(agent)

      if len(actions)==0:
        actions = (gameState,None)

      minMaxFlag = 0
      maxVal = list()
      minVal = list()
      #pacman's turn , if 0, using same minimax algo
      if agent == 0:
        maxVal = [-100000000000.0,None]
        for action in actions:
          nextVal = self.alphaBetaFunction((agent+1)%gameState.getNumAgents(),gameState.generateSuccessor(agent, action),depth,alpha,beta)
          nextVal.append(action)
          maxVal = max(maxVal,nextVal)
          if (maxVal)[0]>beta:
            return maxVal
          
          alpha = max(maxVal[0],alpha)
          minMaxFlag = 1
      #ghost's turn
      else:
        minVal = [100000000000.0,None]
        for action in actions:
          
          nextVal = self.alphaBetaFunction((agent+1)%gameState.getNumAgents(),gameState.generateSuccessor(agent, action),depth + (((agent+1)%gameState.getNumAgents()) == 0),alpha,beta)
          nextVal.append(action)
          minVal = min(minVal,nextVal)
          if minVal[0]<alpha:
            return minVal
          
          beta = min(minVal[0],beta)
          minMaxFlag = 2
      
      if minMaxFlag == 1:
        return maxVal
      else:
        return minVal



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

        expectiMax = self.expectiMaxFunction(0, gameState, 0)
        return expectiMax[-1]

    def expectiMaxFunction(self, agent, gameState, depth):
      
      if depth == self.depth or gameState.isWin() or gameState.isLose():
        ans = list()
        ans.append(self.evaluationFunction(gameState))
        return ans
      
      #keep all the possible states in a list
      possibleStates = list()
      for move in gameState.getLegalActions(agent):
        temp = list()
        temp.append(gameState.generateSuccessor(agent, move))
        temp.append(move)
        possibleStates.append(temp)

      if len(possibleStates) == 0:
        possibleStates = [(gameState, None)]
      
      #if pacman's turn , means agent is 0, we choose the max of all possible
      if agent == 0:
        maxVal = [-100000000]
        
        for gState,action in possibleStates:
          nextVal = self.expectiMaxFunction((agent+1)%gameState.getNumAgents(), gState, depth)
          nextVal.append(action)
          maxVal = max(maxVal,nextVal)
        return (maxVal)

      #else we form the expected value
      else:
        expectedValue = []
        expectedVal = 0
        expectedAction= None
        actions = gameState.getLegalActions(agent)
        for gState,action in possibleStates:
          nextVal = self.expectiMaxFunction((agent+1)%gameState.getNumAgents(), gState, depth + ((agent+1)%gameState.getNumAgents()==0))
          expectedVal += float(nextVal[0])/float(len(actions))
          expectedAction = action
        expectedValue.append(expectedVal)
        expectedValue.append(expectedAction)  
        return (expectedValue)

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

