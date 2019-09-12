import itertools
from operator import itemgetter
import numpy as np
import torch
import pickle
import os
import pdb

class Word_info:
	def __init__(self,vocab_data):

		self.vocab_data=vocab_data
		self.word2index = {"PAD":0,"SOS":1,"EOS":2,"UNK":3,"COPY":4}
		self.word2count = {}
		self.index2word = {0: "PAD", 1: "SOS", 2:"EOS",3:"UNK",4:"COPY"}
		self.n_words = 5  # Count SOS and EOS

	def addWord(self, word):

		if word not in self.word2index:
			self.word2index[word] = self.n_words
			#self.word2count[word] = 1
			self.index2word[self.n_words] = word
			self.n_words += 1
		else:
			self.word2count[word] += 1
		
	def trim(self,vocab_size):
		
		for k in self.vocab_data[0:vocab_size]:
			self.word2count[k[0]]=k[1]
		keep_words=self.word2count.keys()
		for word in keep_words:
			self.addWord(word)

def create_glove_embedding_matrix(word_info,params):

	count=0
	not_in=[]
	embed=torch.randn(word_info.n_words,params['embedding_size'])
	if params['use_glove']:
		w2id=word_info.word2index
		word_dict,w_vec,emb_size=torch.load(params['embed_file_path'])
		for word,ind in w2id.items():
			if word in word_dict.keys():
				vector=w_vec[word_dict[word]]
				embed[ind][:emb_size].copy_(vector)
			else:
				count+=1
				not_in.append(word)
	print(count)
	#print(not_in)
	return embed

def read_data(file_path):

	with open(file_path,'rb') as f:
		data=pickle.load(f)
	print('Data reading complete')
	return data

def make_directory(folder_path):

	if os.path.exists(folder_path)==False:
		os.makedirs(folder_path)