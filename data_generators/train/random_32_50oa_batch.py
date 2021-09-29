import os
import json
import numpy as np
import random
import multiprocessing as mp
import math
import gzip
import argparse

OUT_DIM = 32
PATH_PERC = 0.3
AGENT_PERC = 0.3

def get_normalised_ctg(cost_to_go, max_cost_possible):
  max_current = np.max(cost_to_go[cost_to_go <= max_cost_possible])
  min_current = np.min(cost_to_go)
  if len(cost_to_go[cost_to_go >= max_cost_possible]) != 0:
    new_max = max_current + (max_current - min_current)
  else:
    new_max = max_current
  cost_to_go[cost_to_go >= max_cost_possible] = new_max
  cost_to_go = (cost_to_go - min_current) / (new_max - min_current)
  return cost_to_go

def generate_batches(files, idx, output_dir):
  output_file_num = 0
  out_X = []
  out_y = []
  for file in files:
    try:
      with gzip.open(file, 'rt', encoding='ascii') as fin:
        data = json.load(fin)
    except Exception as e:
      print(file)
      print(e)
      continue
    
    map_dim = data['map_dim']
    num_agents = data['num_agents']
    
    world = np.array(data['world'])
    goals = data['goals']
    all_costs_to_go = data['all_costs_to_go']
    path = data['path']
    
    '''
    1. Obstacle Map - Obstacles marked as 1
    2. Agent Start Map - Initial Position marked as 1
    3. Agent Goal Map - Goal position marked as 1
    4. Cost-to-Map - Smallest value => 0 for goal, rest scaled between 0 and 1
    5. Other Agent Start Map - Initial Positions marked as 1
    6. Other Agent Goal Map - Other goal position marked as 1
    7. Sum of Cost to Map - Other Agents
    8. Future 1 of Other Agents
    9. Future 2 of Other Agents
    10. Future 3 of Other Agents
    '''
    X_base = np.zeros((10, OUT_DIM, OUT_DIM))
              
    X_base[0] = world
    
    path_idxs = [i for i in range(len(path)-1)]
    random.shuffle(path_idxs)
    path_idxs = path_idxs[:int(len(path) * PATH_PERC)]
    
    for path_idx in path_idxs:
      init_pos = path[path_idx]
      next_pos = path[path_idx + 1]
      
      agent_idxs = [i for i in range(num_agents)]
      random.shuffle(agent_idxs)
      agent_idxs = agent_idxs[:int(num_agents * AGENT_PERC)]
      
      for i in agent_idxs:
        X = np.copy(X_base)
        start = init_pos[i]
        end = goals[i]
        
        X[1][start[0]][start[1]] = 1
        X[2][end[0]][end[1]] = 1
        
        X[3] = get_normalised_ctg(np.array(all_costs_to_go[i]), map_dim[0] * map_dim[1])
        
        rows = map_dim[0]
        cols = map_dim[1]
        
        sum_of_costs_to_go = np.zeros((OUT_DIM, OUT_DIM))
        
        for j in range(num_agents):
          if i == j:
            continue
          
          other_start = init_pos[j]
          other_goal = goals[j]
          other_cost_to_go = get_normalised_ctg(np.array(all_costs_to_go[j]), map_dim[0] * map_dim[1])
          
          X[4][other_start[0]][other_start[1]] = 1
          X[5][other_goal[0]][other_goal[1]] = 1
          
          sum_of_costs_to_go += other_cost_to_go
          
          pos_x = other_start[0]
          pos_y = other_start[1]
          
          moves = 3
          array_idx = 7
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
            
        X[6] = get_normalised_ctg(sum_of_costs_to_go, map_dim[0] * map_dim[1])
        
        start_x = start[0]
        start_y = start[1]
        next_pos_x = next_pos[i][0]
        next_pos_y = next_pos[i][1]
        
        if start_x == next_pos_x and start_y == next_pos_y:
          y = 0 #no move
        elif start_x - 1 == next_pos_x and start_y == next_pos_y:
          y = 1 #move up
        elif start_x + 1 == next_pos_x and start_y == next_pos_y:
          y = 2 #move down
        elif start_x == next_pos_x and start_y - 1 == next_pos_y:
          y = 3 #move left
        elif start_x == next_pos_x and start_y + 1 == next_pos_y:
          y = 4 #move right
        else:
          raise Exception('Illegal move')
        
        if len(out_X) % 1000 == 0:
          print(len(out_X))
        
        out_X.append(X)
        out_y.append(y)
      
      if len(out_X) >= 8192:
        rand = np.random.permutation(len(out_X))
        out_X_randomized = [out_X[i] for i in rand]
        out_y_randomized = [out_y[i] for i in rand]
        
        batch_size = 16
        while len(out_X_randomized) >= batch_size:
          X_batch = out_X_randomized[:batch_size]
          y_batch = out_y_randomized[:batch_size]
  
          rand = np.random.permutation(len(X_batch))
          X_batch_randomized = [X_batch[i] for i in rand]
          y_batch_randomized = [y_batch[i] for i in rand]
  
          with open(output_dir + str(idx) + '_' + str(output_file_num) + '.npz', 'wb') as fout:
            np.savez_compressed(fout, X=np.array(X_batch_randomized), y=np.array(y_batch_randomized))
          
          output_file_num += 1
          
          if output_file_num % 10 == 0:
            print('Working on idx:', idx, 'Output file num:', output_file_num)
          
          out_X_randomized = out_X_randomized[batch_size:]
          out_y_randomized = out_y_randomized[batch_size:]
          
        out_X = out_X_randomized
        out_y = out_y_randomized
        
  return idx

def log_result(idx):
    print('Completed:', idx)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-inputdir', required=True)
  parser.add_argument('-outputdir', required=True)
  args = parser.parse_args()
  
  input_dir = args.inputdir
  output_dir = args.outputdir
  
  os.makedirs(output_dir, exist_ok=True)
  
  files = [input_dir + x for x in os.listdir(input_dir)]
  print('Total inputs: ', len(files))
  random.shuffle(files)
  
  threads = 20
  pool = mp.Pool(threads)
  results = []
  
  split_size = math.ceil(len(files)/threads)
  files_split = []
  idx = 0
  while 1:
    files_split.append(files[idx: idx + split_size])
    idx += split_size
    if idx > len(files):
      break

  for i in range(min(threads, len(files_split))):
    future = pool.apply_async(generate_batches, [files_split[i], i, output_dir], callback=log_result)
    results.append(future)
    
  for r in results:
    r.get()
    
    


