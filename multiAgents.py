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
from math import log

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

        "*** YOUR CODE HERE ***"
        #minimum distance from the ghost
        minGhost=float('inf') 
        for state in newGhostStates:
            x= manhattanDistance(newPos, state.getPosition())
            if x<minGhost:
                minGhost=x

        pos= currentGameState.getPacmanPosition()

        #minimum distance from the food
        minFood= min([manhattanDistance(food,pos) for food in currentGameState.getFood().asList()])
        newFood= [manhattanDistance(food,newPos) for food in successorGameState.getFood().asList()]
        if not newFood:
            x= 0
        else:
            x= min(newFood)
        dist= minFood-x     
        
        if minGhost<2 or action== Directions.STOP:
            return -1
        if successorGameState.getScore()-currentGameState.getScore()>0:
            return 10
        elif dist>0:
            return 5
        else:
            return 1        

        
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        actions= gameState.getLegalActions(0)
        best=None
        val=float('-inf')
        for action in actions:
            successor= gameState.generateSuccessor(0,action)
            v= self.min_value(successor,1,0)
            if v>val:
                val=v
                best=action
        
        return best
        
    def max_value(self, gameState,depth):
        v=float('-inf')
        d=depth
        actions= gameState.getLegalActions(0)
        if (len(actions)==0 or d==self.depth):  #terminal state. (max depth or win or lose)
            return self.evaluationFunction(gameState)

        for action in actions:
            successor= gameState.generateSuccessor(0,action)
            val=self.min_value(successor,1,d)
            if val>v:
                v=val
        return v

    def min_value(self, gameState,agentIndex,depth):
        v=float('inf')
        d=depth
        actions= gameState.getLegalActions(agentIndex)
        if (len(actions)==0 or d==self.depth):
            return self.evaluationFunction(gameState)            
        
        
        for action in actions:
            successor= gameState.generateSuccessor(agentIndex,action)
            if (agentIndex+1)%gameState.getNumAgents() ==0: #nextturn is of the Pacman.
                val= self.max_value(successor,d+1)
            else:
                val= self.min_value(successor,agentIndex+1,d)        
            if val<v:
                v=val
        return v        





class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.max_value(gameState,0,float('-inf'),float('inf'))
        
    def max_value(self, gameState,depth,a,b):
        v=float('-inf')
        best=None
        actions= gameState.getLegalActions(0)
        if len(actions)==0 or depth==self.depth:  #terminal state. (max depth or win or lose)
            return self.evaluationFunction(gameState)

        for action in actions:
            successor= gameState.generateSuccessor(0,action)
            val= self.min_value(successor,1,depth,a,b)
            if val>v:
                v=val
                if depth==0 or best==None:
                    best=action
            if v>b:
                break
            a= max(a,v)

        if depth==0:
            return best    

        return v

    def min_value(self, gameState,agentIndex,depth,a,b):
        v=float('inf')
        d=depth
        actions= gameState.getLegalActions(agentIndex)
        if (len(actions)==0 or d==self.depth):
            return self.evaluationFunction(gameState)            
                
        for action in actions:
            successor= gameState.generateSuccessor(agentIndex,action)
            
            if (agentIndex+1)%gameState.getNumAgents() ==0: #nextturn is of the Pacman.
                val= self.max_value(successor,d+1,a,b)
            else:
                val= self.min_value(successor,agentIndex+1,d,a,b)        
            
            if val<v:
                v=val
            if v<a:
                break
            b=min(b,v)

        return v


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
        actions= gameState.getLegalActions(0)
        best=None
        val=float('-inf')
        for action in actions:
            successor= gameState.generateSuccessor(0,action)
            v= self.exp_value(successor,1,0)
            if v>val:
                val=v
                best=action
        return best


    def max_value(self, gameState,depth):
        v=float('-inf')
        d=depth
        actions= gameState.getLegalActions(0)
        if (len(actions)==0 or d==self.depth):  #terminal state. (max depth or win or lose)
            return self.evaluationFunction(gameState)

        for action in actions:
            successor= gameState.generateSuccessor(0,action)
            val=self.exp_value(successor,1,d)
            if val>v:
                v=val
        return v

    def exp_value(self, gameState,agentIndex,depth):
        v=0
        d=depth
        actions= gameState.getLegalActions(agentIndex)
        if (len(actions)==0 or d==self.depth):
            return self.evaluationFunction(gameState)

        p= 1.0/len(actions)             
        
        for action in actions:
            successor= gameState.generateSuccessor(agentIndex,action)
            if (agentIndex+1)%gameState.getNumAgents() ==0: #nextturn is of the Pacman.
                v+= self.max_value(successor,d+1)*p
            else:
                v+= self.exp_value(successor,agentIndex+1,d)*p        
        
        return v




class MCTS(MultiAgentSearchAgent):

    class MCNode:
        def _init_(self, state, parent, action):
            self.state = state
            self.parent = parent
            self.action = action
            self.legalActions = state.getLegalPacmanActions()
            self.visits = 0
            self.score = 0
            self.explored = False
            self.child = []

    def scoreEval(self, state):
        return state.getScore() + [0,-1000.0][state.isLose()] + [0,1000.0][state.isWin()]

    def gameEvaluation(self, startState, currentState):
        rootEval = self.scoreEval(startState);
        currentEval = self.scoreEval(currentState);
        return (currentEval - rootEval) / 1000.0;
    
    def UCT(self,node):
        best = float('-Inf')
        uctNodes = []
        
        for child in node.child:
            childUCT = (child.score/child.visits) + ((log(node.visits))/child.visits)**0.5
            if childUCT >= best:
                if childUCT == best:
                    uctNodes.append(child)
                uctNodes = [child]
                best = childUCT
        return random.choice(uctNodes)
    
    def TreeSearch(self,node):
        while node is not None:
            if node.state.isWin() or node.state.isLose():
                break
            elif node.explored:
                node = self.UCT(node)
            else:
                return self.expand(node)
        return None
    
    def expand(self,node):
        #reduce the possible actions by adding a child node for each action
        rand = random.randint(0,len(node.legalActions)-1)
        action = node.legalActions.pop(rand)
        newState = node.state.generatePacmanSuccessor(action)
        if newState is None:
            self.none = True
        
        newNode = MCNode(newState,node,action)
        node.child.append(newNode)
        
        # check if all the possible actions are taken
        if len(node.legalActions())==0:
            node.explored = True
        
        return newNode
    
    def Policy(self,node):
        state = node.state
        for _ in range(0,5):
            if not state.isWin() and not state.isLose() and state is not None:
                action = random.choice(state.getLegalPacmanActions())
                newState = state.generatePacmanSuccessor(action)
                if newState is None:
                    self.none = True
                    break
                else:
                    state = newState
        
        return self.gameEvaluation(self.rootState,state)
    
    def backPropagation(self, node, evalue):
        while node is not None:
            node.visits += 1
            node.score += evalue
            node = node.parent
            
    def getAction(self, state):
        root = MCTS.MCNode(state, None, None)
        self.none = False
        
        while not self.none:
            new = self.TreeSearch(root)
            if new is not None:
                simulationScore = self.Policy(new)
                self.backPropagation(new, simulationScore)
        
        maxVisits = max([node.visits for node in root.child])
        best = []
        for node in root.childNodes:
            if node.numberOfVisits == maxVisits:
                best.append(node)
        
        rand = random.randint(0,len(best)-1)
        if len(best) == 1:
            return best[0].action
        else:
            return best[rand].action


    