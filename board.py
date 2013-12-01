import pygame
from RL_framework import *

class StateIcon:
    def __init__(self, radius = 50, color = (255,255,255), width = 0, font_size = 80):
        self.radius = radius
        self.color = color
        self.width = width
        self.font = pygame.font.SysFont(pygame.font.get_default_font(), font_size)
        
    def display(self, screen, state):
        x = screen.get_width()/2
        y = screen.get_height()/2
        label = self.font.render(str(state), 1, (0,0,0))
        pygame.draw.circle(screen, self.color, (x, y), self.radius, self.width)
        screen.blit(label, (x-(label.get_width()/2),y-(label.get_height()/2)))


class ActionsIcon:
    def __init__(self, radius = 50, color = (0,0,255), width = 0):
        self.radius = radius
        self.color = color
        self.width = 0
        
    def actionA(self, screen):
        x = screen.get_width()/2
        y = screen.get_height()/2
        r = self.radius
        p1 = (x+(3*r/2), y-r)
        p2 = (x+(3*r/2), y+r)
        p3 = (x+(5*r/2), y)
        pygame.draw.polygon(screen, self.color, [p1, p2, p3], self.width)
        
    def actionB(self, screen):
        x = screen.get_width()/2
        y = screen.get_height()/2
        r = self.radius
        p1 = (x-(3*r/2), y-r)
        p2 = (x-(3*r/2), y+r)
        p3 = (x-(5*r/2), y)
        pygame.draw.polygon(screen, self.color, [p1, p2, p3], self.width)
    
    def actionC(self, screen):
        x = screen.get_width()/2
        y = screen.get_height()/2
        r = self.radius
        p1 = (x-r, y-(3*r/2))
        p2 = (x+r, y-(3*r/2))
        p3 = (x, y-(5*r/2))
        pygame.draw.polygon(screen, self.color, [p1, p2, p3], self.width)
    
    def actionD(self, screen):
        x = screen.get_width()/2
        y = screen.get_height()/2
        r = self.radius
        p1 = (x-r, y+(3*r/2))
        p2 = (x+r, y+(3*r/2))
        p3 = (x, y+(5*r/2))
        pygame.draw.polygon(screen, self.color, [p1, p2, p3], self.width)
        
    def display(self, screen, actions):
        for action in actions:
            if action == 0:
                self.actionA(screen)
            elif action == 1:
                self.actionB(screen)
            elif action == 2:
                self.actionC(screen)
            elif action == 3:
                self.actionD(screen)
        
        
class RewardIcon:
    def __init__(self, start = 0, font_size = 50):
        self.reward = start
        self.font_size = font_size
        self.font = pygame.font.SysFont(pygame.font.get_default_font(), font_size)
        
    def update(self, reward):
        if reward != None:
            self.reward += reward
        
    def display(self, screen):
        pad = self.font_size/5
        x = screen.get_width()
        y = screen.get_height()
        label = self.font.render(str(self.reward), 1, (255,255,255))
        screen.blit(label, (x-label.get_width()-pad,pad))


class Board:
    def __init__(self, model):
        self.model = model
        self.state = StateIcon()
        self.actions = ActionsIcon()
        self.reward = RewardIcon()
        
    def _performAction(self, action):
        reward = None
        actions = self.model.get_actions(self.model.current_state)
        
        if action != None and action in actions:
            reward = self.model.perform(action)
            
        self.reward.update(reward)
            
        return reward
        
    def actionA(self):
        action = self.model.get_action_by_id(0)
        return self._performAction(action)
    
    def actionB(self):
        action = self.model.get_action_by_id(1)
        return self._performAction(action)
    
    def actionC(self):
        action = self.model.get_action_by_id(2)
        return self._performAction(action)
    
    def actionD(self):
        action = self.model.get_action_by_id(3)
        return self._performAction(action)
    
    def display(self, screen):
        self.state.display(screen, self.model.current_state.get_id())
        
        action_ids = []
        for a in self.model.get_actions(self.model.current_state):
            action_ids.append(a.get_id())    
        self.actions.display(screen, action_ids)
        
        self.reward.display(screen)
        
