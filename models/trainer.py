import time
import torch
import os
import numpy as np
import argparse
from basic_block import BasicBlock
from model import ViTResNet, data_loader, train, evaluate

BATCH_SIZE = 64

def setup_train_and_test_file(train_file, test_file):
  print('Setting up train and test file.')
  files = []  
  for path, subdirs, f in os.walk(input_dir):
    for name in f:
      files.append(os.path.join(path, name))
    
  perm = np.random.permutation(len(files))
  files = [files[i] for i in perm]
  
  n = len(files)
  print('Total input files found:', n)
  
  train_size = int(0.9 * n)
  train_files = files[ : train_size]
  test_files = files[train_size :]
  
  with open(train_file, 'w') as fout:
    for file in train_files:
        fout.write(file + '\n')
  
  with open(test_file, 'w') as fout:
      for file in test_files:
          fout.write(file + '\n')

def train_model(input_dir, train_device, epochs, model_path):
  device = torch.device(train_device if torch.cuda.is_available() else "cpu")
  train_file = input_dir + 'train.txt'
  test_file = input_dir + 'test.txt'
  
  if not (os.path.exists(train_file) and os.path.exists(test_file)):
    setup_train_and_test_file(train_file, test_file)
  
  model = ViTResNet(BasicBlock, [3, 3, 3], BATCH_SIZE)
  model.to(device)
  
  optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.992)
  
  train_loader = data_loader(train_file, BATCH_SIZE, device)
  test_loader = data_loader(test_file, BATCH_SIZE, device)
  
  train_loss_history, test_loss_history = [], []
  best_val_loss = 1e9
  for epoch in range(1, epochs + 1):
      print('Epoch:', epoch)
      start_time = time.time()
      train(model, optimizer, train_loader, train_loss_history, scheduler)
      print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
      acc, val_loss = evaluate(model, test_loader, test_loss_history)
      if val_loss < best_val_loss:
        torch.save(model.state_dict(), model_path)
        best_val_loss = val_loss  

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-inputdir', required=True)
  parser.add_argument('-device', required=True)
  parser.add_argument('-epochs', required=True, type=int)
  parser.add_argument('-modelpath', required=True)
  args = parser.parse_args()
  
  input_dir = args.inputdir
  train_device = args.device
  epochs = args.epochs
  model_path = args.modelpath
  
  train_model(input_dir, train_device, epochs, model_path)
  
