import torch
from torch.autograd import Variable
import torch.nn.init as weight_init
import numpy as np
import pdb
import time
import datetime
import os
import sys
from data_utils import *
from data_loader import *
#from params import *
from enc_dec import *
from bleu import *
from rouge import *
from f1score import *
import pandas as pd
from operator import itemgetter
import argparse
import matplotlib.pyplot as plt

torch.manual_seed(2)
#device_id = torch.cuda.current_device()
def get_sequence_loss(pad_ind):

	seq_loss=nn.CrossEntropyLoss(size_average=False,ignore_index=pad_ind,reduce=False)
	return seq_loss

def get_span_loss():

	span_loss=nn.CrossEntropyLoss(size_average=False,reduce=False)
	return span_loss

def criterion(pad_ind):

	seq_loss=get_sequence_loss(pad_ind)
	span_loss=get_span_loss()
	return seq_loss,span_loss

def get_optimizer(model,params_dict):

	parameters_trainable = list(filter(lambda p: p.requires_grad,model.parameters()))
	optimizer=torch.optim.Adam(parameters_trainable,lr=params_dict['learning_rate'])
	return optimizer,parameters_trainable

def save_checkpoint(state_info,checkpoint_path,epoch):

	best_checkpoint='model_best_epoch.pth.tar'
	#check_normal=os.path.join(checkpoint_path,checkpoint_file_name)
	check_best=os.path.join(checkpoint_path,best_checkpoint)
	#if not os.path.isfile(check_best):
	torch.save(state_info,check_best)

def flatten_ip(all_gen_op,decoder_op,p_star_copy,all_p_copy):

	all_gen_op_flat=all_gen_op.contiguous().view(-1,all_gen_op.size(-1))
	decoder_op_flat=decoder_op.contiguous().view(-1,1).squeeze(1)
	p_star_copy_flat=p_star_copy.contiguous().view(p_star_copy.size(0)*p_star_copy.size(1))
	all_p_copy_flat=all_p_copy.contiguous().view(all_p_copy.size(0)*all_p_copy.size(1))
	return all_gen_op_flat,decoder_op_flat,p_star_copy_flat,all_p_copy_flat

def calc_loss(seq_loss,span_loss,all_gen_op,decoder_op,p_star_gen,all_p_gen,p_star_copy,all_p_copy,start_attn_scores,end_attn_scores,batch_span):

	all_gen_op_flat,decoder_op_flat,p_star_copy_flat,all_p_copy_flat=flatten_ip(all_gen_op,decoder_op,p_star_copy,all_p_copy)		
	copy_ind=p_star_copy_flat.nonzero().squeeze(1)
	#start_soft=F.softmax(start_attn_scores,dim=1)
	#end_soft=F.softmax(end_attn_scores,dim=1)
	p_copy_select=all_p_copy_flat.index_select(dim=0,index=copy_ind)
	each_seq_loss=seq_loss(all_gen_op_flat,decoder_op_flat).view(decoder_op.size())
	final_seq_loss=(torch.sum(p_star_gen*each_seq_loss*torch.log(all_p_gen)))/decoder_op.size(1)
	#start_span_loss=span_loss(start_attn_scores,batch_span[:,0])
	#end_span_loss=span_loss(end_attn_scores,batch_span[:,1])
	each_span_loss=(span_loss(start_attn_scores,batch_span[:,0])+span_loss(end_attn_scores,batch_span[:,1]))
	final_span_loss=torch.sum(each_span_loss*torch.log(p_copy_select))
	avg_loss=-(final_seq_loss+final_span_loss)/decoder_op.size(0)
	return avg_loss

def check_resume_model(model,optimizer,best_bleu,start_epoch,flags):

	if flags.resume_path:
		if os.path.isfile(flags.resume_path):
			print("Loading and resuming from checkpoint")
			checkpoint=torch.load(flags.resume_path)
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			for state in optimizer.state.values():
				for k, v in state.items():
					if isinstance(v, torch.Tensor):
						state[k] = v.cuda()
			best_bleu=checkpoint['best_bleu']
			start_epoch=checkpoint['epoch']
	return model,optimizer,best_bleu,start_epoch

def init_params(flags):
    
	params_dict={}
	params_dict['train_file_path']='../final_data/final_train.pkl'
	params_dict['valid_file_path']='../final_data/final_valid.pkl'
	params_dict['test_file_path']='../final_data/final_test.pkl'
	params_dict['vocab_file_path']='../final_data/final_vocab.pkl'
	params_dict['embed_file_path']='../glove_6B/glove.6B.300d.pt'    
	params_dict['checkpoint_path']='checkpoint'
	params_dict['embedding_size']=300
	params_dict['use_glove']=True
	params_dict['vocab_size']=20000    
	params_dict['embed_update']=True
	params_dict['n_layers']=1    
	params_dict['logs_path']='logs'
	params_dict['log_file_name']=flags.log_file_name
	params_dict['learning_rate']=flags.learning_rate
	params_dict['gradient_clip']=flags.gradient_clip
	params_dict['patience']=flags.patience
	params_dict['hidden_size']=flags.hidden_size
	params_dict['num_epochs']=flags.num_epochs
	params_dict['batch_size']=flags.batch_size
	params_dict['prev_connection']=flags.prev_connection
	params_dict['bidirectional']=flags.bidirectional
	params_dict['weight_init']=flags.weight_init   
	params_dict['share_weights']=flags.share_weights 
	if flags.network=='GRU':
		params_dict['network']=nn.GRU
		params_dict['cell_type']=nn.GRUCell
	else:
		params_dict['network']=nn.LSTM
		params_dict['cell_type']=nn.LSTMCell        
	return params_dict        
    
def parse_arguments():

	parser=argparse.ArgumentParser()
	parser.add_argument('--log_file_name',type=str,default='log.txt')
	parser.add_argument('--batch_size',type=int,default=64)
	parser.add_argument('--num_epochs',type=int,default=50)
	parser.add_argument('--hidden_size',type=int,default=128)
	parser.add_argument('--prev_connection',type=bool,default=False)
	parser.add_argument('--bidirectional',type=bool,default=False)
	parser.add_argument('--learning_rate',type=float,default=0.02)
	parser.add_argument('--gradient_clip',type=float,default=5)
	parser.add_argument('--patience',type=int,default=5)
	parser.add_argument('--network',type=str,default='GRU')
	parser.add_argument('--weight_init',type=str,default='random')
	parser.add_argument('--early_stop',type=bool,default=False)  
	parser.add_argument('--resume_path',type=str,default=None)    
	parser.add_argument('--exp_name',type=str,default=None)
	parser.add_argument('--share_weights',type=bool,default=False)
	flags=parser.parse_args()
	return flags

def write_file(file_path,to_write):

	pickle.dump(to_write,open(file_path,'wb'))

def write_details_to_file(train_loss_list,val_loss_list,bleu_list,f1_list,rouge_list,test_str,test_pred_response,test_true_response,test_pred_span,test_true_span,logs_path):

	train_detail={}
	valid_detail={}
	test_detail={}
	train_detail['loss']=train_loss_list
	valid_detail['loss']=val_loss_list
	valid_detail['bleu']=bleu_list
	valid_detail['f1']=f1_list
	valid_detail['rouge']=rouge_list
	test_detail['detail']=test_str
	test_detail['true_response']=test_true_response
	test_detail['pred_response']=test_pred_response
	test_detail['true_span']=test_true_span
	test_detail['pred_span']=test_pred_span
	write_file(os.path.join(logs_path,'./train_detail.pkl'),train_detail)
	write_file(os.path.join(logs_path,'./valid_detail.pkl'),valid_detail)
	write_file(os.path.join(logs_path,'./test_detail.pkl'),test_detail)
	df=pd.DataFrame({'true_response':test_true_response,'pred_response':test_pred_response})
	df.to_csv(os.path.join(logs_path,'./output.csv'),index=False)

def call_init(model,to_choose):

	for name,params in model.named_parameters():
		if 'weight' in name:
			if to_choose=='xavier':
				weight_init.xavier_normal_(params)
			if to_choose=='orthogonal':
				weight_init.orthogonal_(params)
	return model

def perform_init(model,params_dict):

	if params_dict['weight_init']=='xavier':
		model=call_init(model,'xavier')
	elif params_dict['weight_init']=='orthogonal':
		model=call_init(model,'orthogonal')
	return model

def main():

	pdb.set_trace()
	flags=parse_arguments()
	exp_name=flags.exp_name    
	#exp_dict=exp_details()
	#params_dict=exp_dict[exp_name]()   
	params_dict=init_params(flags)    
	print(params_dict,flush=True) 
	logs_path=os.path.join(exp_name,params_dict['logs_path'])
	checkpoint_path=os.path.join(exp_name,params_dict['checkpoint_path'])
	make_directory(logs_path)
	make_directory(checkpoint_path)
	log_file_loc=os.path.join(logs_path,params_dict['log_file_name'])
	sys.stdout=open(log_file_loc,'w')
	vocab_data=pickle.load(open(params_dict['vocab_file_path'],'rb'))
	print("Vocab is created",flush=True)
	word_info=Word_info(vocab_data)
	word_info.trim(params_dict['vocab_size'])
	print("Only Top words in vocab",flush=True)
	word_embedding=create_glove_embedding_matrix(word_info,params_dict)
	print("Glove Embedding matrix created",flush=True)
	train_dataset = DialogDataset(word_info,params_dict['train_file_path'])
	valid_dataset = DialogDataset(word_info,params_dict['valid_file_path'])
	test_dataset = DialogDataset(word_info,params_dict['test_file_path'])

	#make_batch_data(word_info,train_dataset.pairs[256:320],False)
	train_loader=train_dataset.get_dataloader(batch_size=params_dict['batch_size'],shuffle=True)
	valid_loader=valid_dataset.get_dataloader(batch_size=params_dict['batch_size'],shuffle=False)
	test_loader=test_dataset.get_dataloader(batch_size=params_dict['batch_size'],shuffle=False)
	train_data_len=train_dataset.__len__()
	valid_data_len=valid_dataset.__len__()
	test_data_len=test_dataset.__len__()
	#train_max_length=get_max_len(train_dataset,word_info)
	val_max_length,val_original_response,val_original_span=get_true_data(valid_dataset,word_info)
	test_max_length,test_original_response,test_original_span=get_true_data(test_dataset,word_info)
	no_train_batches=train_data_len/params_dict['batch_size']
	no_valid_batches=valid_data_len/params_dict['batch_size']
	no_test_batches=test_data_len/params_dict['batch_size']    
	start_epoch=0
	#best_loss=np.inf
	best_bleu=0    
	model=Model(word_embedding,params_dict)
	model=perform_init(model,params_dict)
	print("Model is initialized",flush=True)
	seq_loss,span_loss=criterion(word_info.word2index['PAD'])
	optimizer,parameters_trainable=get_optimizer(model,params_dict)
	model,optimizer,best_bleu,start_epoch=check_resume_model(model,optimizer,best_bleu,start_epoch,flags)
	if torch.cuda.is_available():
		model=model.cuda()
	best_epoch=start_epoch-1
	bleu_list=[]
	f1_list=[]
	val_loss_list=[]
	train_loss_list=[]
	rouge_list=[]
	mode='train'
	for epoch in range(start_epoch,params_dict['num_epochs']):
		start_time=datetime.datetime.now().replace(microsecond=0)
		loss_accross_epoch=0
		for batch_no,train_batch in enumerate(train_loader):
			optimizer.zero_grad()
			query_batch,query,history_batch,history,document_batch,document,response_batch,response,batch_span,true_copy_gen=train_batch
			p_star_gen,p_star_copy=true_copy_gen
			decoder_op=response_batch[:,1:]
			print("Training started",flush=True)
			all_gen_op,start_attn_scores,end_attn_scores,all_p_gen,all_p_copy=model(word_info,mode,query_batch,query,history_batch,history,document_batch,document,response_batch,response,p_star_copy)
			avg_loss=calc_loss(seq_loss,span_loss,all_gen_op,decoder_op,p_star_gen,all_p_gen,p_star_copy,all_p_copy,start_attn_scores,end_attn_scores,batch_span)
			print("step %d of epoch %d " %(batch_no,epoch),flush=True)
			print("Train Loss: ",avg_loss.cpu().data.numpy(),flush=True)
			avg_loss.backward()
			#pa=parameters_trainable[0]
			#print(pa.grad)
			'''for name,pa in model.named_parameters():
				if bool(torch.isnan(pa.grad).any()):
					print(name)
					break'''
			loss_accross_epoch+=avg_loss.cpu().data.numpy()
			nn.utils.clip_grad_norm_(parameters_trainable,params_dict['gradient_clip'])
			'''for name,pa in model.named_parameters():
				if bool(torch.isnan(pa.grad).any()):
					print(name)'''
			optimizer.step()        
		elapsed_time=datetime.datetime.now().replace(microsecond=0)-start_time
		print("Elapsed time in an epoch is ",elapsed_time,flush=True)
		print("epoch %d Train loss %f " %(epoch,loss_accross_epoch/no_train_batches),flush=True)
		val_loss,val_pred_response,val_pred_span=perform_eval(seq_loss,span_loss,valid_loader,word_info,val_max_length,model,params_dict,no_valid_batches)
		val_bleu_score,val_rouge_dict,val_f1=evaluate(val_original_response,val_pred_response)
		print("epoch %d Overall Validation Loss %f " %(epoch,val_loss),flush=True)
		print("epoch %d Validation F1 score %f " %(epoch,val_f1),flush=True)
		print("epoch %d Validation Bleu score %f " %(epoch,val_bleu_score),flush=True)
		print("epoch is ",epoch,flush=True)
		print("Validation rouge scores are ",val_rouge_dict,flush=True)
		test_loss,test_pred_response,test_pred_span=perform_eval(seq_loss,span_loss,test_loader,word_info,test_max_length,model,params_dict,no_test_batches)        
		test_bleu_score,test_rouge_dict,test_f1=evaluate(test_original_response,test_pred_response)
		print("epoch %d Overall Test Loss %f " %(epoch,test_loss),flush=True)
		print("epoch %d Test F1 score %f " %(epoch,test_f1),flush=True)
		print("epoch %d Test Bleu score %f " %(epoch,test_bleu_score),flush=True)
		print("epoch is ",epoch,flush=True)
		print("Test rouge scores are ",test_rouge_dict,flush=True)
		train_loss_list.append(loss_accross_epoch/no_train_batches)
		val_loss_list.append(val_loss)
		bleu_list.append(val_bleu_score)
		f1_list.append(val_f1)
		rouge_list.append(val_rouge_dict)               
		'''if val_loss<best_loss:
			best_loss=val_loss
			best_epoch=epoch
			print("best_epoch is ",best_epoch,flush=True)
			print("Saving checkpoint",flush=True)'''
		if val_bleu_score>best_bleu:
			best_bleu=val_bleu_score
			best_epoch=epoch
			print("best_epoch is ",best_epoch,flush=True)
			print("Saving checkpoint",flush=True)
			save_checkpoint({'epoch':epoch+1,'state_dict':model.state_dict(),'best_bleu':best_bleu,'optimizer':optimizer.state_dict()},checkpoint_path,epoch)

		#if flags.early_stop==True and (epoch-best_epoch)>params_dict['patience']:
			#print("Validation bleu does not increase from last "+str(params_dict['patience'])+" epochs. So stopped training",flush=True)
			#break
	print("Loading best model for testing",flush=True)
	best_model_path=os.path.join(checkpoint_path,'model_best_epoch.pth.tar')
	checkpoint=torch.load(best_model_path)
	model.load_state_dict(checkpoint['state_dict'])    
	test_loss,test_pred_response,test_pred_span=perform_eval(seq_loss,span_loss,test_loader,word_info,test_max_length,model,params_dict,no_test_batches)
	test_bleu_score,test_rouge_dict,test_f1=evaluate(test_original_response,test_pred_response)
	print("Test Loss ",test_loss,flush=True)
	print("Test F1 score ",test_f1,flush=True)
	print("Test Bleu score is ",test_bleu_score,flush=True)
	print("Test rouge scores are ",test_rouge_dict,flush=True)
	test_str='Test loss is '+str(test_loss)+'\n'+'Test F1 score is '+str(test_f1)+'\n'+'Test Bleu score is '+str(test_bleu_score)+'\n'+'Test rouge score is '+str(test_rouge_dict)
	write_details_to_file(train_loss_list,val_loss_list,bleu_list,f1_list,rouge_list,test_str,test_pred_response,test_original_response,test_pred_span,test_original_span,logs_path)

def get_true_data(dataset,word_info):

	resp_list=[]
	original_resp=[]
	original_span=[]
	data=dataset.pairs
	for each_sample in data:
		replaced_response=add_copy_to_response(lower_case(each_sample['response']),lower_case(each_sample['span']))
		response='SOS '+replaced_response+' EOS'
		resp_list.append(response)
		original_resp.append(' '.join([token.text for token in tokenize.tokenizer(lower_case(each_sample['response'])) if token.text!=' ']))
		original_span.append(' '.join([token.text for token in tokenize.tokenizer(lower_case(each_sample['span'])) if token.text!=' ']))
		#original_resp.append(' '.join(tokenize(lower_case(each_sample['response']))))
		#original_span.append(' '.join(tokenize(lower_case(each_sample['span']))))
	ops=basic_ops(resp_list,word_info)
	indices=ops.get_batch_data_index()
	lengths=[len(seq) for seq in indices]
	max_length=max(lengths)
	return max_length,original_resp,original_span

def get_pred_span_sentences(start_max_ind,end_max_ind,document_batch,word_info):

	batch_pred_span=[]
	for step,(start,end) in enumerate(zip(start_max_ind.cpu().data.tolist(),end_max_ind.cpu().data.tolist())):
		if end>start:
			#pred_span_ind=document_batch[step][start:end+1]
			pred_span_ind=document_batch[step][start:end+1].data.tolist()
			batch_pred_span.append(' '.join([word_info.index2word[each_word_ind] for each_word_ind in pred_span_ind]))
			#batch_pred_span.append(' '.join(list(itemgetter(*pred_span_ind.data.tolist())(word_info.index2word))))
		else:
			batch_pred_span.append(' ')
	return batch_pred_span

def replace_response_by_span(batch_resp_words,batch_pred_span,copy_ind):

	for each_ind in copy_ind:
		pos1,pos2=each_ind
		if pos1<len(batch_resp_words) and pos2<len(batch_resp_words[pos1]):
			batch_resp_words[pos1][pos2]=batch_pred_span[pos1]

def get_pred_batch_resp_sentences(all_top_ind,batch_pred_span,copy_ind,word_info):

	batch_resp_words=[]
	for seq_top_ind in all_top_ind.cpu().data.tolist():
		to_find=word_info.word2index['EOS']
		if to_find in seq_top_ind:
			upto_index=seq_top_ind.index(to_find)
			batch_resp_words.append([word_info.index2word[each_top_ind] for each_top_ind in seq_top_ind[0:upto_index]])
			#batch_resp_words.append(list(itemgetter(*seq_top_ind[0:upto_index])(word_info.index2word)))	
		else:
			batch_resp_words.append([word_info.index2word[each_top_ind] for each_top_ind in seq_top_ind])
			#batch_resp_words.append(list(itemgetter(*seq_top_ind)(word_info.index2word)))
	replace_response_by_span(batch_resp_words,batch_pred_span,copy_ind)
	return batch_resp_words

def calc_bleu(true_response,pred_response):

	bleu_score=moses_multi_bleu(true_response,pred_response)
	return bleu_score

def calc_rouge(true_response,pred_response):

	rouge_dict=rouge(true_response,pred_response)
	return rouge_dict

def evaluate(true_response,pred_response):

	total_f1=0
	bleu_score=calc_bleu(true_response,pred_response)
	rouge_dict=calc_rouge(true_response,pred_response)
	for each_true_resp,each_pred_resp in zip(true_response,pred_response):
		total_f1+=f1_score(each_pred_resp,each_true_resp)
	total_f1=(total_f1*100)/len(true_response)  
	return bleu_score,rouge_dict,total_f1

def perform_eval(seq_loss,span_loss,data_loader,word_info,max_length,model,params_dict,no_batches):
	
	print("Evaluation started",flush=True)
	model.eval()
	mode='eval'
	loss_accross_epoch=0
	all_resp_sent=[]   
	pred_span_list=[]
	for step,data_batch in enumerate(data_loader):
		with torch.no_grad(): 
			query_batch,query,history_batch,history,document_batch,document,response_batch,response,batch_span,true_copy_gen=data_batch
			if max_length>response_batch.size(1):
				to_pad=max_length-response_batch.size(1)
				zero_var=Variable(torch.zeros(response_batch.size(0),to_pad))
				if torch.cuda.is_available():
					zero_var=zero_var.cuda()
				pad_var=zero_var.clone().fill_(word_info.word2index['PAD']).long()
				decoder_op=torch.cat((response_batch[:,1:],pad_var),dim=1)
				p_star_gen=torch.cat((true_copy_gen[0],zero_var),dim=1)
				p_star_copy=torch.cat((true_copy_gen[1],zero_var),dim=1)
			else:
				decoder_op=response_batch[:,1:]
			all_gen_op,start_attn_scores,end_attn_scores,start_max_ind,end_max_ind,all_p_gen,all_p_copy,all_top_ind=model(word_info,mode,query_batch,query,history_batch,history,document_batch,document,response_batch,response,max_length-1)
			avg_loss=calc_loss(seq_loss,span_loss,all_gen_op,decoder_op,p_star_gen,all_p_gen,p_star_copy,all_p_copy,start_attn_scores,end_attn_scores,batch_span)
			loss_accross_epoch+=avg_loss.cpu().data.numpy()
			print("batch_no %d Inference Loss %f " %(step,avg_loss),flush=True)
			doc_original_order=document.get_original_order(document_batch)
			batch_pred_span=get_pred_span_sentences(start_max_ind,end_max_ind,doc_original_order,word_info)
			all_comp_prob=all_p_copy>all_p_gen
			copy_ind=all_comp_prob.data.nonzero().cpu().data.tolist()
			batch_resp_words=get_pred_batch_resp_sentences(all_top_ind,batch_pred_span,copy_ind,word_info)
			for span in batch_pred_span:
				pred_span_list.append(span)
			for sent_words in batch_resp_words:
				all_resp_sent.append(' '.join(sent_words))
	model.train()
	return loss_accross_epoch/no_batches,all_resp_sent,pred_span_list

if __name__=="__main__":

	main()
