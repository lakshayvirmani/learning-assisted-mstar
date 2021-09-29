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
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import argparse
import resource

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
      if path[j][i] == path[j-1][i]:
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

def solve_mapf_instances(inputs, output_dir, inflation):
  track_failed = dict()
  for file in inputs: 
    print('Working on file: ', file)
    
    resource.setrlimit(resource.RLIMIT_AS, (2**34, 2**34))
    
    try:
        data = json.load(open(file, 'r'))
    except:
        continue
    
    world = np.array(data['world'])
    obs = data['obs']
    init_pos = data['init_pos']
    goals = data['goals']
    file_name = file.split('/')[-1]
    num_agents = len(data['init_pos'])
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
  
    try:
      m_output = od_mstar.find_path(world, init_pos, goals, astar=True, recursive=False, inflation=inflation, time_limit=300, connect_8=False, return_memory=True)
      mac = np.sum(get_multi_agent_costs_to_go(m_output['path'], goals))
      num_unique_collision_coords, num_agents_colliding = extract_collision_information(m_output['collisions'])
      
      print('Sum of Costs: ', mac)
      print('Time taken: ', m_output['time_taken'])
      print('Nodes Popped: ', m_output['nodes_popped'])
      print('Graph Size: ', m_output['corrected_graph_size'])
      print('Max collision set size: ', m_output['max_collision_set_size'])
      print('No. of unique collision coords: ', num_unique_collision_coords)
      print('No. of unique agents colliding: ', num_agents_colliding)
      print('No. of total collisions: ', m_output['total_collisions'])
      
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
  
    m_output['mac'] = mac
    m_output['num_unique_collision_coords'] = num_unique_collision_coords
    m_output['num_agents_colliding'] = num_agents_colliding
    
    output = {}
    
    output['map_dim'] = data['map_dim']
    output['num_agents'] = num_agents
    output['obs'] = obs
    output['init_pos'] = init_pos
    output['goals'] = goals
    output['m_output'] = m_output
    
    json.dump(output, open(out_file, 'w'))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-inputdir', required=True)
  parser.add_argument('-outputdir', required=True)
  parser.add_argument('-inflation', type=float, required=True)
  args = parser.parse_args()
  
  input_dir = args.inputdir
  output_dir = args.outputdir
  inflation = args.inflation
  
  os.makedirs(output_dir, exist_ok=True)
  
  inputs = [input_dir + x for x in os.listdir(input_dir)]
  print('Total Inputs: ', len(inputs))
  inputs = sorted(inputs, key=sort_key)

  solve_mapf_instances(inputs, output_dir, inflation)