import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from functools import partial
from data_utils import read_data
from nltk import word_tokenzie,sent_tokenize
#import spacy
import pdb

#tokenize=spacy.load('en',parser=False, tagger=False,entity=False)

class DialogDataset(Dataset):

	def __init__(self, word_info, path):

		super(DialogDataset, self).__init__()
		self.word_info = word_info
		self.pairs = read_data(path)

	def __getitem__(self, idx):
		return self.pairs[idx]

	def __len__(self):
	   return len(self.pairs)	

	def _create_collate_fn(self, batch_first=True):
		
		def collate(examples, this):

			batch_pairs = examples
			query_batches,query,history_batches,history,document_batches,document,response_batches,response,span_batches,true_copy_gen = make_batch_data(self.word_info, batch_pairs)
			return query_batches,query,history_batches,history,document_batches,document,response_batches,response,span_batches,true_copy_gen
		return partial(collate, this=self)   

	def get_dataloader(self, batch_size, num_workers=0, shuffle=False, batch_first=True, pin_memory=False):
		"""

		:param batch_first:  Currently, it must be True as nn.Embedding requires batch_first input
		"""
		batch_obj=DataLoader(self, batch_size=batch_size, shuffle=shuffle,collate_fn=self._create_collate_fn(batch_first),num_workers=num_workers, pin_memory=pin_memory)
		return batch_obj

class basic_ops():

	def __init__(self,batch_list,word_info):

		self.batch_list=batch_list
		self.word_info=word_info
		self.batch_to_ind=[]

	'''def pre_process_sent(self,sentence):

		#word_tokens=tokenize(sentence)
		word_tokens=[token.text for token in tokenize.tokenizer(sentence) if token.text!=' ']
		return word_tokens'''

	def tokenize(self,text):

		return list(_tokenize_gen(text))

	def _tokenize_gen(self,text):

		for sent in nltk.sent_tokenize(text):
			for word in nltk.word_tokenize(sent):
				word = word.replace("''", '"').replace("``", '"')
				yield word

	def create_masks(self):

		self.masks = torch.zeros(self.batch_to_ind.size())
		for i,length in enumerate(self.lengths):
			self.masks[i][:length].fill_(1)

	def create_dynamic_masks(self,dynamic_len):

		for i,length in enumerate(dynamic_len):
			self.masks[i][:length+1].fill_(0)

	def pad_seq(self):

		self.lengths=[len(seq) for seq in self.batch_to_ind]
		max_length=max(self.lengths)
		pad_ind=self.word_info.word2index['PAD']
		for seq in self.batch_to_ind:
			#pad_seq(seq,max_length,pad_ind)
			seq += [pad_ind for i in range(max_length - len(seq))]

	def sort_in_desc(self):

		sorted_lengths,self.sorted_id=torch.sort(self.lengths,dim=0,descending=True)
		self.sorted_lengths=list(sorted_lengths)
		self.sorted_tensor=self.batch_to_ind.index_select(dim=0,index=self.sorted_id)
		self.original_id=torch.LongTensor(self.sorted_id.size())
		sorted_id_list=list(self.sorted_id)
		for i in range(len(sorted_id_list)):
			self.original_id[i]=sorted_id_list.index(i)

	def get_batch_data_index(self):

		for each_sample in self.batch_list:
			sent_to_ind=[]
			for word in self.tokenize(each_sample):
				if word in self.word_info.word2index:
					sent_to_ind.append(self.word_info.word2index[word])
				else:
					sent_to_ind.append(self.word_info.word2index["UNK"])
			self.batch_to_ind.append(sent_to_ind)
		return self.batch_to_ind

	def convert_variable(self):

		self.batch_to_ind=Variable(self.batch_to_ind)
		self.sorted_id=Variable(self.sorted_id)
		self.original_id=Variable(self.original_id)
		self.sorted_tensor=Variable(self.sorted_tensor)
		self.masks=Variable(self.masks)

	def convert_torch_tensor(self):

		self.batch_to_ind=torch.LongTensor(self.batch_to_ind)
		self.lengths=torch.LongTensor(self.lengths)

	def convert_to_cuda(self):

		self.batch_to_ind=self.batch_to_ind.cuda()
		self.sorted_id=self.sorted_id.cuda()
		self.original_id=self.original_id.cuda()
		self.sorted_tensor=self.sorted_tensor.cuda()
		self.masks=self.masks.cuda()

	def pad_mask_sort(self):

		self.pad_seq()
		self.convert_torch_tensor()
		self.create_masks()
		self.sort_in_desc()
		self.convert_variable()
		if torch.cuda.is_available():
			self.convert_to_cuda()
		return self.batch_to_ind,self.sorted_tensor

	def get_original_order(self,sorted_seq):

		orig_seq=sorted_seq.index_select(dim=0,index=self.original_id)
		return orig_seq

	def get_to_sorted_order(self,orig_seq):

		to_sorted_seq=orig_seq.index_select(dim=0,index=self.sorted_id)
		return to_sorted_seq

def get_start_end_indices(batch_documents, batch_span, batch_span_start):
	
	batch_documents_token = []
	batch_span_ind=[]
	count = 0
	bad_examples = []
	
	for step,(document, span, span_start) in enumerate(zip(batch_documents, batch_span, batch_span_start)):
		words = []
		doc_tokenize=tokenize(document)
		fore = document[:span_start]
		mid = document[span_start: span_start + len(span)]
		after = document[span_start + len(span):]
		tokenized_fore = tokenize(fore)
		tokenized_mid = tokenize(mid)
		tokenized_after = tokenize(after)
		tokenized_text = tokenize(span)
		for i, j in zip(tokenized_text, tokenized_mid):
			if i != j:
				count = count + 1
				bad_examples.append([tokenized_text,tokenized_mid])
				print("Bad examples are")
				print(bad_examples)
		words.extend(tokenized_fore)
		words.extend(tokenized_mid)
		words.extend(tokenized_after)
		answer_start_token, answer_end_token = len(tokenized_fore), len(tokenized_fore) + len(tokenized_mid) - 1
		batch_documents_token.append(words)
		if answer_end_token>=len(doc_tokenize):
			batch_span_ind.append([answer_start_token,len(doc_tokenize)-1])
		else:        
			batch_span_ind.append([answer_start_token,answer_end_token])
	if torch.cuda.is_available():
		return Variable(torch.LongTensor(batch_span_ind)).cuda()
	else:	
		return Variable(torch.LongTensor(batch_span_ind))

def get_true_copy_gen(batch_res_seqs,response,word_info):

	copy_ind=[]
	true_gen=response.masks[:,1:]
	true_copy=torch.zeros(true_gen.size())
	to_find=word_info.word2index['COPY']
	for each_res_seqs in batch_res_seqs:
		copy_ind.append(each_res_seqs.index(to_find)-1)
	for i in range(true_copy.size(0)):
		true_copy[i][copy_ind[i]]=1
	if torch.cuda.is_available():
		return [true_gen,Variable(true_copy).cuda()]
	else:	
		return [true_gen,Variable(true_copy)]

def do_sort_mask_pad(query,history,document,response):

	_,query_sorted_seqs=query.pad_mask_sort()
	_,history_sorted_seqs=history.pad_mask_sort()
	_,document_sorted_seqs=document.pad_mask_sort()
	response_original_seqs,_=response.pad_mask_sort()
	return query_sorted_seqs,history_sorted_seqs,document_sorted_seqs,response_original_seqs

def add_copy_to_response(cur_response,cur_span):

	replaced_response=cur_response.replace(cur_span,' COPY ')
	return replaced_response

def lower_case(sent):

	return sent.lower()

def make_batch_data(word_info,batch_pairs):
	
	#pdb.set_trace()
	hist_list=[]
	query_list=[]
	doc_list=[]
	resp_list=[]
	span_list=[]
	start_list = []
	batch_span_ind=[]
	for each_sample in batch_pairs:
		query_list.append(lower_case(each_sample['query']))
		hist_list.append(lower_case(' '.join(each_sample['short_history'])))
		doc_list.append(lower_case(each_sample['oracle_reduced']))
		start_list.append(each_sample['answer_start_oracle_reduced'])
		replaced_response=add_copy_to_response(lower_case(each_sample['response']),lower_case(each_sample['span']))
		response='SOS '+replaced_response+' EOS'
		resp_list.append(response)
		span_list.append(lower_case(each_sample['span']))
		batch_span_ind.append(each_sample['start_end_ind'])
	query=basic_ops(query_list,word_info)
	batch_query_seqs=query.get_batch_data_index()
	history=basic_ops(hist_list,word_info)
	batch_history_seqs=history.get_batch_data_index()
	#batch_span_ind=get_start_end_indices(doc_list,span_list,start_list)
	document=basic_ops(doc_list,word_info)
	batch_doc_seqs=document.get_batch_data_index()
	response=basic_ops(resp_list,word_info)
	batch_res_seqs=response.get_batch_data_index()
	span=basic_ops(span_list,word_info)
	batch_span_seqs=span.get_batch_data_index()
	query_sorted_seqs,history_sorted_seqs,document_sorted_seqs,response_original_seqs=do_sort_mask_pad(query,history,document,response)
	if torch.cuda.is_available():
		batch_span_ind=Variable(torch.LongTensor(batch_span_ind)).cuda()
	else:
		batch_span_ind=Variable(torch.LongTensor(batch_span_ind))
	true_copy_gen=get_true_copy_gen(batch_res_seqs,response,word_info)
	#if mode=='train':
	return query_sorted_seqs,query,history_sorted_seqs,history,document_sorted_seqs,document,response_original_seqs,response,batch_span_ind,true_copy_gen