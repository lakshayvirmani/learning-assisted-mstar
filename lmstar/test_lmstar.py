#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 18:11:26 2021

@author: lakshayvirmani
"""
import os
import od_mstar
import numpy as np
import json
import traceback
import argparse

def sort_key(file):
  file_name = file.split('/')[-1]
  num_agents = int(file_name.split('_')[-1].split('.')[0])
  base_name = file_name.split('_' + file_name.split('_')[-1])[0]
  return (base_name, num_agents)

def get_multi_agent_costs_to_go(path, goals):
  n = len(goals)
  mac = np.zeros(n)
  
  for i in range(n):
    cost = 0
    for j in range(1, len(path)):
      if path[j][i] == goals[i]:
          break
      if j != 0 and path[j][i] == path[j-1][i]:
        continue
      cost += 1
    
    mac[i] = cost
  
  return mac

def extract_collision_information(collisions):
  coords = set()
  agents = set()
  for col in collisions:
    coords.add(col[0])
    agents.update(col[1])
    
  return len(coords), len(agents)

def solve_mapf_instances(inputs, output_dir, model_dir, trained_model, inflation):
  track_failed = dict()
  for file in inputs:
    print('Working on file: ', file)
    
    try:
        data = json.load(open(file, 'r'))
    except:
        continue
    
    map_dim = data['map_dim']
    world = np.array(data['world'])
    obs = data['obs']
    num_agents = len(data['init_pos'])
    file_name = file.split('/')[-1]
    base_name = file_name.split('_' + file_name.split('_')[-1])[0]
    out_file = output_dir + file_name
    
    if os.path.exists(out_file) or os.path.exists(out_file + '.json'):
      print('Already done.')
      continue
    
    assert len(data['init_pos']) == len(data['goals']) == int(file_name.split('_')[-1].split('.')[0])
    
    print('Base Name: ', base_name)
    
    if base_name in track_failed:
      if num_agents > track_failed[base_name]:
        print('Skipping on {} agents as it failed on {} agents before.'.format(num_agents, track_failed[base_name]))
        continue
    
    model_config = {}
    model_config['model_loc'] = model_dir
    model_config['model_weights'] = trained_model
    model_config['out_dim'] = 32
    model_config['batch_size'] = 64
    model_config['input_channels'] = 10
    
    model_data = {}
    model_data['obs'] = obs
    
    init_pos = data['init_pos']
    goals = data['goals']
    all_costs_to_go = np.array(data['all_costs_to_go'])
    
    model_data['init_pos'] = init_pos
    model_data['goals'] = goals
    model_data['all_costs_to_go'] = all_costs_to_go
    model_data['map_dim'] = map_dim
    model_data['world'] = world
    
    model_config['data'] = model_data
    
    try:
      lm_output = od_mstar.find_path(world, init_pos, goals, model_config, astar=True, recursive=False, inflation=inflation, time_limit=300, connect_8=False, return_memory=True)
      mac = np.sum(get_multi_agent_costs_to_go(lm_output['path'], goals))
      num_unique_collision_coords, num_agents_colliding = extract_collision_information(lm_output['collisions'])
      
      print('Sum of Costs: ', mac)
      print('Time taken: ', lm_output['time_taken'])
      print('Nodes Popped: ', lm_output['nodes_popped'])
      print('Graph Size: ', lm_output['corrected_graph_size'])
      print('Max collision set size: ', lm_output['max_collision_set_size'])
      print('No. of unique collision coords: ', num_unique_collision_coords)
      print('No. of unique agents colliding: ', num_agents_colliding)
      print('No. of total collisions: ', lm_output['total_collisions'])
      
      if base_name in track_failed and track_failed[base_name] < num_agents + 5:
          print('Giving a buffer of 10 agents: ', num_agents + 5)
          track_failed[base_name] = num_agents + 5
      
    except Exception:
      traceback.print_exc()
      if base_name not in track_failed:
          track_failed[base_name] = num_agents + 5
      else:
          track_failed[base_name] = min(num_agents + 5, track_failed[base_name])
      print('Failure tracking: ', track_failed[base_name])
      continue
    
    lm_output['mac'] = mac
    lm_output['num_unique_collision_coords'] = num_unique_collision_coords
    lm_output['num_agents_colliding'] = num_agents_colliding
    
    output = {}
    
    output['map_dim'] = map_dim
    output['num_agents'] = num_agents
    output['obs'] = obs
    output['init_pos'] = init_pos
    output['goals'] = goals
    output['m_output'] = lm_output
    
    json.dump(output, open(out_file, 'w'))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-inputdir', required=True)
  parser.add_argument('-outputdir', required=True)
  parser.add_argument('-modeldir', required=True)
  parser.add_argument('-trainedmodel', required=True)
  parser.add_argument('-inflation', type=float, required=True)
  args = parser.parse_args()
  
  input_dir = args.inputdir
  output_dir = args.outputdir
  model_dir = args.modeldir
  trained_model = args.trainedmodel
  inflation = args.inflation
  
  os.makedirs(output_dir, exist_ok=True)
  
  inputs = [input_dir + x for x in os.listdir(input_dir)]
  print('Total Inputs: ', len(inputs))
  inputs = sorted(inputs, key=sort_key)
  
  solve_mapf_instances(inputs, output_dir, model_dir, trained_model, inflation)
