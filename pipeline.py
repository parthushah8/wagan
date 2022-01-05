# importing libraries
import matplotlib.pyplot as plt
import json
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets
import torchvision.transforms as transforms
# importing from other files
from models import  MNIST_target_net

class TargetModel_Pipeline:
  
  def __init__(self, device, config, MODEL_DIR):
    # initiating variables of a new class
    
    self.device = device
    self.config = config
    
    self.MODEL_DIR = MODEL_DIR
    
    self.model = MNIST_target_net().to(self.device)

  def fit(self, train_dataloader):
    # training the model on the given train dataset
    
    # train_dataloader = DataLoader(mnist_train, batch_size=self.config['batch_size'], shuffle=False, num_workers=1)

    self.model.train()
    opt_model = torch.optim.Adam(self.model.parameters(), lr=self.config['early_learning_rate'])
  
    epochs = self.config['total_epochs']
    loss_epochs = []
    for epoch in range(epochs):
      loss_epoch = 0
      if epoch == self.config['cutoff_epoch']:
        opt_model = torch.optim.Adam(self.model.parameters(), lr=self.config['late_learning_rate'])
      for train_imgs, train_labels in tqdm(train_dataloader):
          train_imgs, train_labels = train_imgs.to(self.device), train_labels.to(self.device)
          logits_model = self.model(train_imgs)
          loss_model = F.cross_entropy(logits_model, train_labels)
          loss_epoch += loss_model
          opt_model.zero_grad()
          loss_model.backward()
          opt_model.step()
      print('loss in epoch %d: %f' % (epoch, loss_epoch.item()))
      loss_epochs.append(loss_epoch.item())
  
    # save the trained model
    torch.save(self.model.state_dict(), self.MODEL_DIR + '/model_' + str(self.config['name']) + '.pth')
    
    # save the config of the trained model
    with open(self.MODEL_DIR + '/config_' + str(self.config['name']) + '.json', 'w') as config_file:
        config_file.write(json.dumps(self.config))
    
    return loss_epochs
    
  def eval(self, test_dataloader):
    # evaluate the trained model on the given test dataset
    
    # test_dataloader = DataLoader(mnist_test, batch_size=self.config['batch_size'], shuffle=True, num_workers=1)
    num_correct = 0
    for test_img, test_label in tqdm(test_dataloader):
      test_img, test_label = test_img.to(self.device), test_label.to(self.device)
      pred_label = torch.argmax(self.model(test_img), 1)
      num_correct += torch.sum(pred_label==test_label,0)
  
    print('accuracy in testing set: %f\n'%(num_correct.item()/len(test_dataloader.dataset)))
    return num_correct.item()/len(test_dataloader.dataset)

  def load(self, config_name):
    # load the config and model corresponding to the congif name
    
    with open(self.MODEL_DIR + '/config_' + str(config_name) + '.json') as config_file:
        self.config = json.loads(config_file.read())

    self.model.load_state_dict(torch.load(self.MODEL_DIR + '/model_' + str(config_name) + '.pth'))

    return
