#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pacman import Directions
from game import Agent
import numpy as np
import random
import game
import util


# In[ ]:


def scoreEval(state):
    return state.getScore() + [0,-1000.0][state.isLose()] + [0,1000.0][state.isWin()]

def gameEvaluation(startState, currentState):
    rootEval = scoreEval(startState);
    currentEval = scoreEval(currentState);
    return (currentEval - rootEval) / 1000.0;


# In[ ]:


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


# In[ ]:


class MCTS(Agent):
    
    def UCT(self,node):
        best = float('-Inf')
        uctNodes = []
        
        for child in node.child:
            childUCT = (child.score/child.visits) + ((np.log(node.visits))/child.visits)**0.5
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
        if simulation in range(0,5):
            if not state.isWin() and not state.isLose() and state is not None:
                action = random.choice(state.getLegalPacmanActions())
                newState = state.generatePacmanSuccessor(action)
                if newState is None:
                    self.none = True
                    break
                else:
                    state = newState
        
        return gameEvaluation(self.rootState,state)
    
    def backPropagation(self, node, evalue):
        while node is not None:
            node.visits += 1
            node.score += evalue
            node = node.parent
            
    def getAction(self, state):
        root = MCNode(state, None, None)
        self.none = False
        
        while not none:
            new = self.TreeSearch(root)
            if new is not None:
                simualtionScore = self.Policy(new)
                self.backPropagation(new, simulationScore)
        
        maxVisits = max([node.visits for node in root.child])
        best = []
        for node in rootNode.childNodes:
            if node.numberOfVisits == maxNumberOfVisits:
                best.append(node)
        
        rand = random.randint(0,len(best)-1)
        if len(best) == 1:
            return best[0].action
        else:
            return best[rand].action        

