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

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    
    #A list to keep track of visited states.
    visited = []
    #Creating a stack of states to do bfs.
    stateStack = util.Stack()
    #Push the start state
    stateStack.push([(problem.getStartState(),"Stop",0)])
    
    #Iterate through the queue till either the stack is not empty or we haven't reached the goal state
    while not stateStack.isEmpty():
        pacway = stateStack.pop()
        
        curr = pacway[len(pacway)-1]
        curr = curr[0]
        
        #If the current state is goal exit and return
        if problem.isGoalState(curr):
            return [p[1] for p in pacway][1:]
        
        #If the current state is not a goal mark it as visited and check for all it's successor
        #If the successors are not visited put them in the stack.
        if curr not in visited:
            visited.append(curr)
            
            for succ in problem.getSuccessors(curr):
                if succ[0] not in visited:
                    succPath = pacway[:]
                    succPath.append(succ)
                    stateStack.push(succPath)
            
    return []
    

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first.
    
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    "*** YOUR CODE HERE ***"
    """

    #A list to keep track of visited states.
    visited = []
    #Creating a queue of states to do bfs.
    stateQueue = util.Queue()
    #Push the start state
    stateQueue.push([(problem.getStartState(), "Stop", 0)])
    
    #Iterate through the queue till either the queue is not empty or we haven't reached the goal state
    while not stateQueue.isEmpty():
        pacway = stateQueue.pop()
        
        curr = pacway[len(pacway)-1]
        curr = curr[0]
        
        #If the current state is goal exit and return
        if problem.isGoalState(curr):
            return [p[1] for p in pacway][1:]
        
        #If the current state is not a goal mark it as visited and check for all it's successor
        #If the successors are not visited put them in the queue and mark them visited.
        visited.append(curr)
        for succ in problem.getSuccessors(curr):
            if succ[0] not in visited:
                visited.append(succ[0])
                succPath = pacway[:]
                succPath.append(succ)
                stateQueue.push(succPath)
            
    return []
    

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    costFunction = lambda path: problem.getCostOfActions([x[1] for x in path][1:])
    
    #A list to keep track of visited states.
    visited = []
    #Creating a queue of states to do BFS.
    statePriorityQueue = util.PriorityQueueWithFunction(costFunction)
    
    statePriorityQueue.push([(problem.getStartState(),"Stop",0)])
    
    #Iterate through the queue till either the queue is not empty or we haven't reached the goal state
    while not statePriorityQueue.isEmpty():
        pacway = statePriorityQueue.pop()
        
        curr = pacway[len(pacway)-1]
        curr = curr[0]

        #If the current state is goal exit and return
        if problem.isGoalState(curr):
            return [p[1] for p in pacway][1:]
        
        #If the current state is not a goal mark it as visited and check for all it's successor
        #If the successors are not visited put them in the queue.
        if curr not in visited:
            visited.append(curr)       
            for succ in problem.getSuccessors(curr):
                if succ[0] not in visited:
                    succPath = pacway[:]
                    succPath.append(succ)
                    statePriorityQueue.push(succPath)
            
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
    #print problem.getCostOfActions([x[1] for x in path][1:])
    costFunction = lambda path: problem.getCostOfActions([x[1] for x in path][1:]) + heuristic(path[-1][0], problem)
                                                
    #A list to keep track of visited states.
    visited = []
    #Creating a queue of states to do BFS.
    stateQueue = util.PriorityQueueWithFunction(costFunction)
    
    stateQueue.push([(problem.getStartState(),"Stop",0)])
    
    #Iterate through the queue till either the queue is not empty or we haven't reached the goal state
    while not stateQueue.isEmpty():
        pacway = stateQueue.pop()
        
        curr = pacway[len(pacway)-1]
        curr = curr[0]

        #If the current state is goal exit and return
        if problem.isGoalState(curr):
            return [p[1] for p in pacway][1:]
        
        #If the current state is not a goal mark it as visited and check for all it's successor
        #If the successors are not visited put them in the queue.
        if curr not in visited:
            visited.append(curr)
            for succ in problem.getSuccessors(curr):
                if succ[0] not in visited:
                    succPath = pacway[:]
                    succPath.append(succ)
                    stateQueue.push(succPath)
            
    return []
    
                                                    
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
