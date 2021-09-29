#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 21:14:50 2021

@author: lakshayvirmani
"""
import sys
import torch
import numpy as np
import torch.nn.functional as F

#torch.set_num_threads(20)

moves = {
0: (0, 0), #wait
1: (-1, 0), #up
2: (1, 0), #down
3: (0, -1), #left
4: (0, 1) #right
}

class Predictor:
  def __init__(self, model_config):
    self.data = model_config['data']
    self.model_loc = model_config['model_loc']
    self.model_weights = model_config['model_weights']
    self.out_dim = model_config['out_dim']
    self.input_channels = model_config['input_channels']
    self.batch_size = model_config['batch_size']
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    self.obs = [(x[0], x[1]) for x in self.data['obs']]
    self.goals = [(x[0], x[1]) for x in self.data['goals']]
    self.all_costs_to_go = np.array(self.data['all_costs_to_go'])
    self.map_dim = self.data['map_dim']
    self.world = np.array(self.data['world'])
    
    sys.path.append(self.model_loc)
    from model import ViTResNet 
    from basic_block import BasicBlock
    
    self.net = ViTResNet(BasicBlock, [3, 3, 3], self.batch_size)
    self.net.to(self.device)
    self.net.load_state_dict(torch.load(self.model_weights, map_location=torch.device(self.device)))
    #switch model to eval mode to disable dropout
    self.net.eval()
    
    self.model_input = np.ones((self.batch_size, self.input_channels, self.out_dim, self.out_dim), dtype=float)
    
    self.already_predicted = dict()
    
  def prediction_already_done(self, init_pos):
    return init_pos in self.already_predicted
    
  def get_normalised_and_padded_ctg(self, cost_to_go, max_cost_possible, top_pad, bottom_pad, left_pad, right_pad):
    max_current = np.max(cost_to_go[cost_to_go <= max_cost_possible])
    min_current = np.min(cost_to_go)
    if len(cost_to_go[cost_to_go >= max_cost_possible]) != 0:
      new_max = max_current + (max_current - min_current)
    else:
      new_max = max_current
    cost_to_go[cost_to_go >= max_cost_possible] = new_max
    cost_to_go = np.copy(np.pad(cost_to_go, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant', constant_values=new_max))
    cost_to_go = (cost_to_go - min_current) / (new_max - min_current)
    return cost_to_go
  
  def prepare_input_data(self, init_pos, model_input, input_channels, out_dim, map_dim, world, goals, all_costs_to_go):
    top_pad = np.floor((out_dim - map_dim[0]) / 2).astype(int)
    bottom_pad = np.ceil((out_dim - map_dim[0]) / 2).astype(int)
    right_pad = np.ceil((out_dim - map_dim[1]) / 2).astype(int)
    left_pad = np.floor((out_dim - map_dim[1]) / 2).astype(int)
    
    bottom_pad += top_pad
    right_pad += left_pad
    top_pad = 0
    left_pad = 0
    
    world_padded = np.copy(np.pad(world, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant', constant_values=1))
    
    X_base = np.zeros((input_channels, out_dim, out_dim))
    X_base[0] = world_padded
    
    num_agents = len(init_pos)
    
    for i in range(num_agents):
      X = np.copy(X_base)
      start = init_pos[i]
      end = goals[i]
      
      X[1][start[0]][start[1]] = 1
      X[2][end[0]][end[1]] = 1
      
      X[3] = self.get_normalised_and_padded_ctg(np.array(all_costs_to_go[i]), map_dim[0] * map_dim[1], top_pad, bottom_pad, left_pad, right_pad)
      
      rows = map_dim[0]
      cols = map_dim[1]
      
      sum_of_costs_to_go = np.zeros((out_dim, out_dim))
      
      for j in range(num_agents):
        if i == j:
          continue
        
        other_start = init_pos[j]
        other_goal = goals[j]
        other_cost_to_go = self.get_normalised_and_padded_ctg(np.array(all_costs_to_go[j]), map_dim[0] * map_dim[1], top_pad, bottom_pad, left_pad, right_pad)
        
        X[4][other_start[0]][other_start[1]] = 1
        X[5][other_goal[0]][other_goal[1]] = 1
        
        sum_of_costs_to_go += other_cost_to_go
        
        pos_x = other_start[0]
        pos_y = other_start[1]
        
        moves = 3
        
        if input_channels == 9:
          array_idx = 6
        elif input_channels == 10:
          array_idx = 7
        else:
          raise Exception('Unidentified no of input channels found.')
        
        while moves > 0:
          min_x = pos_x
          min_y = pos_y
          min_cost = map_dim[0] * map_dim[1]
          
          if pos_x > 0 and other_cost_to_go[pos_x - 1][pos_y] < min_cost:
            min_cost = other_cost_to_go[pos_x - 1][pos_y]
            min_x = pos_x - 1
            min_y = pos_y
          if pos_x < rows - 1 and other_cost_to_go[pos_x + 1][pos_y] < min_cost:
            min_cost = other_cost_to_go[pos_x + 1][pos_y]
            min_x = pos_x + 1
            min_y = pos_y 
          if pos_y > 0 and other_cost_to_go[pos_x][pos_y - 1] < min_cost:
            min_cost = other_cost_to_go[pos_x][pos_y - 1]
            min_x = pos_x
            min_y = pos_y - 1 
          if pos_y < cols - 1 and other_cost_to_go[pos_x][pos_y + 1] < min_cost:
            min_cost = other_cost_to_go[pos_x][pos_y + 1]
            min_x = pos_x
            min_y = pos_y + 1
          
          X[array_idx][min_x][min_y] = 1
          
          pos_x = min_x
          pos_y = min_y
          
          moves -= 1
          array_idx += 1
      
      if input_channels == 10:
        X[6] = self.get_normalised_and_padded_ctg(sum_of_costs_to_go, map_dim[0] * map_dim[1], 0, 0, 0, 0)
      
      model_input[i] = np.copy(X)
      
  def predict(self, init_pos):
    model_input = np.copy(self.model_input)
    input_channels = self.input_channels
    out_dim = self.out_dim
    map_dim = self.map_dim
    world = self.world
    goals = self.goals
    all_costs_to_go = self.all_costs_to_go
    device = self.device
    
    self.prepare_input_data(init_pos, model_input, input_channels, out_dim, map_dim, world, goals, all_costs_to_go)
    
    output = F.log_softmax(self.net(torch.from_numpy(model_input).float().to(device)), dim=1)
    
    _, indices = list(torch.sort(output, dim=1, descending=True))
    #print(indices[: len(init_pos)])
    
    rows = map_dim[0]
    cols = map_dim[1]
    
    new_coords = []
    
    for i in range(len(init_pos)):
      start_x = init_pos[i][0]
      start_y = init_pos[i][1]
      """
      if start_x == goals[i][0] and start_y == goals[i][1]:
        new_coords.append((start_x, start_y))
        continue
      """
      for move in indices[i]:
        new_x = start_x + moves[move.item()][0]
        new_y = start_y + moves[move.item()][1]
        
        if (new_x, new_y) in self.obs:
          continue
        
        if new_x >= 0 and new_x < rows and new_y >= 0 and new_y < cols:
          new_coords.append((new_x, new_y))
          break
    
    self.already_predicted[tuple(init_pos)] = tuple(new_coords)
    
    return new_coords
