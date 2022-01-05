from tqdm.notebook import tqdm
import itertools

import os
import json
import matplotlib.pyplot as plt

import torch.nn as nn
import torch
import numpy as np
import models
from wasserstein import RegularizedWassersteinDistance
import torch.nn.functional as F
import torchvision
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# custom weights initialization called on netGen and netDisc
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
class WAGAN_Attack:
  
  def __init__(self, device, config):
    
    
    self.config = config
    self.config_name = config['name']
    self.device = device
    self.model_num_labels = config['model_num_labels']
    self.input_nc = config['image_nc']
    self.output_nc = config['image_nc']
    self.box_min = config['box_min']
    self.box_max = config['box_max']

    target_model = models.MNIST_target_net().to(self.device)
    target_model.load_state_dict(torch.load(config['trained_model']))
    target_model.eval()
    self.model = target_model

    self.gen_input_nc = config['image_nc']
    netGen = models.Generator(self.gen_input_nc, config['image_nc']).to(self.device)
    netDisc = models.Discriminator(config['image_nc']).to(self.device)

    # Loss parameters
    self.adv_loss_fnc = config['adv_loss_fnc']
    self.adv_lambda = config['adv_lambda']
    self.pert_lambda = config['pert_lambda']

    self.WassersteinDistance = RegularizedWassersteinDistance(28, device=device)

    # initialize or load weights
    if config['weights'] == 'init':
      netGen.apply(weights_init)
      netDisc.apply(weights_init)
    elif config['weights'] == 'load':
      netGen.load_state_dict(torch.load(config['trained_gen_model']))
      netDisc.load_state_dict(torch.load(config['trained_disc_model']))
    self.netGen = netGen
    self.netDisc = netDisc

    # Learning Rates after cutoff epochs
    self.initial_lr = config['initial_lr']
    self.cutoff_epochs1 = config['cutoff_epochs1']
    self.cutoff_lr1 = config['cutoff_lr1']
    self.cutoff_epochs2 = config['cutoff_epochs2']
    self.cutoff_lr2 = config['cutoff_lr2']

    self.optimizer_Gen = torch.optim.Adam(self.netGen.parameters(), lr=self.initial_lr)
    self.optimizer_Disc = torch.optim.Adam(self.netDisc.parameters(), lr=self.initial_lr)


  def train_batch(self, batch_x, batch_labels):
    
    # Generate Adverserial Images - used Clipping Trick
    perturbation = self.netGen(batch_x)
    adv_images = torch.clamp(perturbation, -0.3, 0.3) + batch_x
    adv_images = torch.clamp(adv_images, self.box_min, self.box_max)


    # Optimize Discriminator
    self.optimizer_Disc.zero_grad()

    # Disc loss on real image
    pred_real = self.netDisc(batch_x)
    loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device))
    loss_D_real.backward()

    # Disc loss on fake(adveserial) image
    pred_fake = self.netDisc(adv_images.detach())
    loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device))
    loss_D_fake.backward()
    
    # Gradient Descent Step
    loss_D_GAN = loss_D_fake + loss_D_real
    self.optimizer_Disc.step()


    # Optimize Generator
    self.optimizer_Gen.zero_grad()

    # Gen loss as per GAN
    pred_fake = self.netDisc(adv_images)
    loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device=self.device))
    loss_G_fake.backward(retain_graph=True)

    # Target model loss on adverserial example
    logits_model = self.model(adv_images)
    probs_model = F.softmax(logits_model, dim=1)
    onehot_labels = torch.eye(self.model_num_labels, device=self.device)[batch_labels]
    # mse, cross-entropy or C&W loss function
    if self.adv_loss_fnc == 'mse_loss':
      loss_adv = -F.mse_loss(logits_model, onehot_labels)
    elif self.adv_loss_fnc == 'cross_entropy':
      loss_adv = -F.cross_entropy(logits_model, labels)
    else:
      real = torch.sum(onehot_labels * probs_model, dim=1)
      other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
      zeros = torch.zeros_like(other)
      loss_adv = torch.max(real - other, zeros)
      loss_adv = torch.sum(loss_adv)

    # Perturbation L2 norm loss 
    loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))
    # loss_perturb = torch.max(loss_perturb - 0.1, torch.zeros(1, device=self.device))
    
    # Wasserstien Loss
    loss_W = self.WassersteinDistance(batch_x, adv_images.detach())

    # New Gen Loss apart from GAN 
    loss_G = self.adv_lambda * loss_adv + self.pert_lambda * loss_perturb + loss_W
    loss_G.backward()

    # Gradient Descent Step
    self.optimizer_Gen.step()

    return loss_D_GAN.item(), loss_G_fake.item(), loss_adv.item(), loss_perturb.item(), loss_W.item()


  def fit(self, train_dataloader, epochs, WAGAN_DIR):

    loss_D_epoch, loss_G_fake_epoch, loss_adv_epoch, loss_perturb_epoch, loss_W_epoch = [], [], [], [], []
    os.makedirs(WAGAN_DIR + '/' + str(self.config_name), exist_ok=True)

    for epoch in range(1, epochs+1):
      
      if epoch == self.cutoff_epochs1:
        self.optimizer_Gen = torch.optim.Adam(self.netGen.parameters(), lr=self.cutoff_lr1)
        self.optimizer_Disc = torch.optim.Adam(self.netDisc.parameters(), lr=self.cutoff_lr1) 
      if epoch == self.cutoff_epochs2:
        self.optimizer_Gen = torch.optim.Adam(self.netGen.parameters(), lr=self.cutoff_lr2)
        self.optimizer_Disc = torch.optim.Adam(self.netDisc.parameters(), lr=self.cutoff_lr2)

      loss_D_sum = 0
      loss_G_fake_sum = 0
      loss_perturb_sum = 0
      loss_adv_sum = 0
      loss_W_sum = 0

      for images, labels in tqdm(train_dataloader):
        images, labels = images.to(self.device), labels.to(self.device)
        loss_D_batch, loss_G_fake_batch, loss_adv_batch, loss_perturb_batch, loss_W_batch = self.train_batch(images, labels)
        loss_D_sum, loss_G_fake_sum, loss_adv_sum, loss_perturb_sum, loss_W_sum = loss_D_sum+loss_D_batch, loss_G_fake_sum+loss_G_fake_batch, loss_adv_sum+loss_adv_batch, loss_perturb_sum+loss_perturb_batch, loss_W_sum+loss_W_batch
        
      # print average losses after each epoch
      loss_D_avg, loss_G_fake_avg, loss_adv_avg, loss_perturb_avg, loss_W_avg = loss_D_sum/len(train_dataloader), loss_G_fake_sum/len(train_dataloader), loss_adv_sum/len(train_dataloader), loss_perturb_sum/len(train_dataloader), loss_W_sum/len(train_dataloader)
      print("epoch %d:\n loss_D: %.3f,\n loss_G_fake: %.3f,\n loss_adv: %.3f,\n loss_perturb: %.3f, \n loss_wasserstien: %.3f, \n" % (epoch, loss_D_avg, loss_G_fake_avg, loss_adv_avg, loss_perturb_avg, loss_W_avg))

      loss_D_epoch.append(loss_D_avg)
      loss_G_fake_epoch.append(loss_G_fake_avg)
      loss_adv_epoch.append(loss_adv_avg)
      loss_perturb_epoch.append(loss_perturb_avg)
      loss_W_epoch.append(loss_W_avg)

      # save generator
      if epoch%5==0:
          netGen_file_name = WAGAN_DIR + '/' + str(self.config_name) + '/Gen_epoch_' + str(epoch) + '.pth'
          torch.save(self.netGen.state_dict(), netGen_file_name)

          netDisc_file_name = WAGAN_DIR + '/' + str(self.config_name) + '/Disc_epoch_' + str(epoch) + '.pth'
          torch.save(self.netDisc.state_dict(), netDisc_file_name)
    
    # save the config of the model
    with open(WAGAN_DIR + '/' + str(self.config_name) + '/config.json', 'w') as config_file:
        config_file.write(json.dumps(self.config))
    
    return loss_D_epoch, loss_G_fake_epoch, loss_adv_epoch, loss_perturb_epoch, loss_W_epoch

  def generate_adverserial_images(self, batch_x):
    # Generate Adverserial Images - used Clipping Trick
    perturbation = self.netGen(batch_x)
    batch_adv_x = torch.clamp(perturbation, -0.3, 0.3) + batch_x
    batch_adv_x = torch.clamp(batch_adv_x, self.box_min, self.box_max)
    return batch_adv_x

  def calc_adverserial_acc(self, dataloader):
    
    num_correct_orig, num_correct_adv = 0, 0
    for images, labels in tqdm(dataloader):
      images, labels = images.to(self.device), labels.to(self.device)
      # Generate adverserial images
      adv_images = self.generate_adverserial_images(images).to(self.device)
      # Get real and adverserial images accuracy
      pred_lab_orig = torch.argmax(self.model(images), 1)
      num_correct_orig += torch.sum(pred_lab_orig==labels,0)
      pred_lab_adv = torch.argmax(self.model(adv_images), 1)
      num_correct_adv += torch.sum(pred_lab_adv==labels,0)

    return num_correct_orig.item(), num_correct_adv.item()