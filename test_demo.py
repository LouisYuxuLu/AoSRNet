# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 20:48:05 2020

@author: Administrator
"""


import torch
import torch.nn as nn

#from tensorboardX import SummaryWriter
import numpy as np
import cv2
import time
import os
from AoSRNet import *
import utils_train


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_checkpoint(checkpoint_dir):
    
	model_info = torch.load(checkpoint_dir + 'checkpoint.pth.tar')
	net = AoSRNet()
	device_ids = [0]
	model = nn.DataParallel(net, device_ids=device_ids).cuda()
	model.load_state_dict(model_info['state_dict'])
	optimizer = torch.optim.Adam(model.parameters())
	optimizer.load_state_dict(model_info['optimizer'])
	cur_epoch = model_info['epoch']

	return model



def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])

def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0])
	

if __name__ == '__main__': 	
	checkpoint_dir = './checkpoint/'
	test_dir = './input'
	result_dir = './output'    
	testfiles = os.listdir(test_dir)
    
	IsGPU = 0    #GPU is 1, CPU is 0

	print('> Loading dataset ...')

	model = load_checkpoint(checkpoint_dir)


	for f in range(len(testfiles)):
		model.eval()
		with torch.no_grad():
			img_c = cv2.imread(test_dir + '/' + testfiles[f]) / 255.0
			img_l = hwc_to_chw(np.array(img_c).astype('float32'))
			input_var = torch.from_numpy(img_l.copy()).type(torch.FloatTensor).unsqueeze(0).cuda()
			s = time.time()
			E_out = model(input_var)
			e = time.time()   
			print(input_var.shape)       
			print('GPUTime:%.4f'%(e-s))    
			E_out = chw_to_hwc(E_out.squeeze().cpu().detach().numpy())			               
			cv2.imwrite(result_dir + '/' + testfiles[f][:-4] + '_AoSRNet.png',np.clip(E_out*255,0.0,255.0))


                
	  
				
			
			

