import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
import torch.nn as nn
from data_utils import *
from attention import *
import pdb

class Word_embedding(nn.Module):

	def __init__(self,word_embedding,params_dict):

		super(Word_embedding,self).__init__()
		self.word_embedding=word_embedding
		self.word_embed_matrix=nn.Embedding(self.word_embedding.size(0),self.word_embedding.size(1))
		self.word_embed_matrix.weight=nn.Parameter(self.word_embedding,requires_grad=params_dict['embed_update'])

	def forward(self,batch_data):

		batch_embedding=[]
		for data in batch_data:
			embed_data=self.word_embed_matrix(data)
			batch_embedding.append(embed_data)
		return batch_embedding

class Rnn_cell(nn.Module):

	def __init__(self,cell_type,input_size,hidden_size):

		super(Rnn_cell,self).__init__()
		self.cell_type=cell_type
		self.input_size=input_size
		self.hidden_size=hidden_size
		self.init_cell=self.cell_type(input_size,hidden_size)

	def forward(self,input_data,prev_hidden):
		
		if self.cell_type==nn.GRUCell:
			next_h=self.init_cell(input_data,prev_hidden)
		else:
			if torch.cuda.is_available():
				c0=Variable(torch.zeros(input_data.size(0),self.hidden_size)).cuda()
			else:
				c0=Variable(torch.zeros(input_data.size(0),self.hidden_size))
			next_h,next_c=self.init_cell(input_data,(prev_hidden,c0))
		return next_h

class Rnn(nn.Module):

	def __init__(self,input_size,hidden_size,n_layers,bidirectional,network):

		super(Rnn,self).__init__()
		self.network=network
		self.net_type=self.network(input_size=input_size,hidden_size=hidden_size,num_layers=n_layers,bidirectional=bidirectional,batch_first=True)

	def forward(self,pack_input):

		if self.network==nn.GRU:
			output,h_t=self.net_type(pack_input)
		else:
			output,(h_t,c_t)=self.net_type(pack_input)
		return output,h_t

class Sentence_encoder(nn.Module):

	def __init__(self,input_size,params):

		super(Sentence_encoder,self).__init__()
		self.share_weights=params['share_weights']
		if self.share_weights:
			self.shared_rnn=Rnn(input_size,params["hidden_size"],params["n_layers"],params["bidirectional"],params["network"])
		else:
			self.query_rnn=Rnn(input_size,params["hidden_size"],params["n_layers"],params["bidirectional"],params["network"])
			self.doc_rnn=Rnn(input_size,params["hidden_size"],params["n_layers"],params["bidirectional"],params["network"])
			self.context_rnn=Rnn(input_size,params["hidden_size"],params["n_layers"],params["bidirectional"],params["network"])

	def forward(self,query_ip,doc_ip,context_ip):

		if self.share_weights:
			query_op,query_ht=self.shared_rnn(query_ip)
			doc_op,doc_ht=self.shared_rnn(doc_ip)
			context_op,context_ht=self.shared_rnn(context_ip)
		else:			
			query_op,query_ht=self.query_rnn(query_ip)
			doc_op,doc_ht=self.doc_rnn(doc_ip)
			context_op,context_ht=self.context_rnn(context_ip)
		return query_op,query_ht,doc_op,doc_ht,context_op,context_ht

class Context_encoder(nn.Module):

	def __init__(self,encode_size,params,to_concat):

		super(Context_encoder,self).__init__()
		self.attn_size=to_concat*encode_size
		self.context_attn=Attention(self.attn_size,params['hidden_size'])

	def forward(self,context_hidden,prev_dec_state,attn_masks):

		qry_aw_ctxt_repr,qry_aw_ctxt_wts,qry_aw_ctxt_scores=self.context_attn(prev_dec_state,context_hidden,attn_masks)
		return qry_aw_ctxt_repr,qry_aw_ctxt_wts,qry_aw_ctxt_scores

class Document_encoder(nn.Module):

	def __init__(self,encode_size,params,to_concat):

		super(Document_encoder,self).__init__()
		self.attn_size=to_concat*encode_size#(2*context or doc_hidden+query)
		self.doc_attn=Attention(self.attn_size,params['hidden_size'])

	def forward(self,doc_hidden,context_repr,query_repr,attn_masks):

		prev_state=torch.cat((context_repr,query_repr),dim=1)
		ctxt_qry_aw_doc_repr,ctxt_qry_aw_doc_wts,ctxt_qry_aw_doc_scores=self.doc_attn(prev_state,doc_hidden,attn_masks)
		return ctxt_qry_aw_doc_repr,ctxt_qry_aw_doc_wts,ctxt_qry_aw_doc_scores

class Copy_gen(nn.Module):

	def __init__(self,encode_size,to_concat):
		#Ask sir for gen_net
		super(Copy_gen,self).__init__()
		self.gen_net=nn.Linear(to_concat*encode_size,1)#(2*context or doc_hidden+resp_dec_state)
		#self.V=nn.Parameter(torch.FloatTensor(1,self.hidden_size))

	def forward(self,context,context_aw_doc,hidden_state):
		
		gen_score=self.gen_net(torch.cat((context,context_aw_doc,hidden_state),1))
		gen_prob=F.sigmoid(gen_score)
		copy_prob=1-gen_prob
		return gen_prob,copy_prob

class Span_prediction(nn.Module):

	def __init__(self,params,input_size,to_concat):

		super(Span_prediction,self).__init__()
		self.span_rnn=Rnn(input_size,params["hidden_size"],params["n_layers"],params["bidirectional"],params["network"])
		#self.one_step_cell=self.cell_type(params["hidden_size"])
		self.no_directions=2 if params['bidirectional']==True else 1
		self.encoding_size=self.no_directions*params['hidden_size']
		self.one_step_cell=Rnn_cell(params['cell_type'],self.encoding_size,(to_concat-1)*self.encoding_size)
		self.attn_size=to_concat*self.encoding_size
		self.attn=Attention(self.attn_size,params["hidden_size"])

	def forward(self,doc_op_original,start_ctxt_repr,start_attn_weights,start_max_ind,context_repr_original,query_repr_original,document):
		
		hidden=doc_op_original.size(2)
		prev_state=torch.cat((context_repr_original,query_repr_original),dim=1)
		next_h=self.one_step_cell(start_ctxt_repr,prev_state)
		#start_max,start_max_ind=torch.max(start_attn_weights,dim=1)
		#document.create_dynamic_masks(start_max_ind)
		attn_wts=start_attn_weights.repeat(hidden,1,1).transpose(0,1).transpose(1,2)
		attn_wt_doc=attn_wts*doc_op_original
		attn_wt_doc_sorted=document.get_to_sorted_order(attn_wt_doc)
		attn_wt_doc_sorted_pack=pack_padded_sequence(attn_wt_doc_sorted,document.sorted_lengths,batch_first=True)
		span_op,_=self.span_rnn(attn_wt_doc_sorted_pack)
		span_op_unpack,_=pad_packed_sequence(span_op,batch_first=True)
		span_op_original=document.get_original_order(span_op_unpack)
		context,end_attn_wts,end_attn_scores=self.attn(next_h,span_op_original,document.masks)
		#context,end_attn_wts,end_attn_scores=self.attn(prev_state,span_op_original,document.masks)
		return end_attn_scores
		
class Gen_span_decoder(nn.Module):

	def __init__(self,params,encoding_size,input_size,to_concat):

		super(Gen_span_decoder,self).__init__()
		self.cell_type=params['cell_type']
		self.encoding_size=encoding_size
		self.one_step_cell=Rnn_cell(params['cell_type'],input_size+to_concat*self.encoding_size,self.encoding_size) #Check input_size
		self.out=nn.Linear(self.encoding_size,params["vocab_size"])

	def forward(self,batch_data,context_repr,doc_repr,h0):

		input_data=torch.cat((batch_data,context_repr,doc_repr),1)
		next_h=self.one_step_cell(input_data,h0)
		output_scores=self.out(next_h)
		output_prob=F.softmax(output_scores,dim=1)
		return output_scores,output_prob,next_h

class Model(nn.Module):
	
	def __init__(self,embedding,params_dict):

		super(Model,self).__init__()
		self.word_embed=Word_embedding(embedding,params_dict)
		self.input_size=embedding.size(1)
		#self.batch_size=params_dict['batch_size']
		self.vocab_size=params_dict['vocab_size']
		self.prev_connection=params_dict['prev_connection']
		self.sentence_enc=Sentence_encoder(self.input_size,params_dict)
		self.no_directions=2 if params_dict['bidirectional']==True else 1
		self.encoding_size=self.no_directions*params_dict['hidden_size']
		self.ctxt_concat_vec=2
		self.context_enc=Context_encoder(self.encoding_size,params_dict,self.ctxt_concat_vec)
		self.doc_concat_vec=3#query and context concat
		self.doc_enc=Document_encoder(self.encoding_size,params_dict,self.doc_concat_vec)
		self.gen_copy_prob=Copy_gen(self.encoding_size,self.doc_concat_vec)
		self.span_pred=Span_prediction(params_dict,self.encoding_size,self.doc_concat_vec)
		self.decoder=Gen_span_decoder(params_dict,self.encoding_size,self.input_size,self.ctxt_concat_vec)

	def forward(self,word_info,mode,*input_tensor):

		if mode=='train':
			teacher_forcing=True
			query_batch,query,context_batch,context,doc_batch,document,response_batch,response,p_star_copy=input_tensor
			decoder_ip_batch=response_batch[:,:-1]
		else:
			teacher_forcing=False
			query_batch,query,context_batch,context,doc_batch,document,response_batch,respons,target_length=input_tensor
			decoder_ip_batch=response_batch[:,0]
		query_embed,context_embed,doc_embed,decoder_ip_embed=self.word_embed([query_batch,context_batch,doc_batch,decoder_ip_batch])
		query_encoding,query_repr,doc_encoding,doc_ht,context_encoding,context_ht=self.passage_context_encode(query_embed,query,doc_embed,document,context_embed,context)
		doc_unpack,_=pad_packed_sequence(doc_encoding,batch_first=True)
		context_unpack,_=pad_packed_sequence(context_encoding,batch_first=True)
		query_repr,doc_ht=self.check_direction(query_repr,doc_ht)
		query_repr_original=query.get_original_order(query_repr)
		doc_padded_original=document.get_original_order(doc_unpack)
		context_padded_original=context.get_original_order(context_unpack)
		context_repr_original,context_attn_wts,context_attn_scores=self.context_enc(context_padded_original,query_repr_original,context.masks)
		doc_repr_original,start_attn_wts,start_attn_scores=self.doc_enc(doc_padded_original,context_repr_original,query_repr_original,document.masks)
		start_max_score,start_max_ind=torch.max(start_attn_scores,dim=1)
		end_attn_scores=self.span_pred(doc_padded_original,doc_repr_original,start_attn_wts,start_max_ind,context_repr_original,query_repr_original,document)
		end_max_score,end_max_ind=torch.max(end_attn_scores,dim=1)
		hidden_state=doc_ht#In place operation check
		if teacher_forcing==True:
			all_gen_op,all_gen_prob,all_copy_prob,add_ind=self.create_batch_variable(decoder_ip_batch.size(1),doc_batch.size(1),mode,decoder_ip_batch.size(0))
			for step,batch_data in enumerate(decoder_ip_embed.transpose(0,1)):
				decoder_step_ip=batch_data                
				if self.prev_connection==True and step-1>=0:
					to_put,non_zero_ind=self.get_prev_connection(p_star_copy.transpose(0,1)[step-1],doc_batch,end_max_ind,add_ind)
					if non_zero_ind.size()!=torch.LongTensor().size() and to_put.size()!=torch.LongTensor().size():
						last_word_embed=self.word_embed([to_put.squeeze(1)])[0]
						decoder_step_ip.data.index_copy_(0,non_zero_ind.data,last_word_embed.data)
				gen_prob,copy_prob=self.gen_copy_prob(context_repr_original,doc_repr_original,hidden_state)
				output_scores,output_prob,hidden_state=self.decoder(decoder_step_ip,context_repr_original,doc_repr_original,hidden_state)
				all_gen_op[step]=output_scores
				all_gen_prob[step]=gen_prob.squeeze(1)
				all_copy_prob[step]=copy_prob.squeeze(1)
			return all_gen_op.transpose(0,1),start_attn_scores,end_attn_scores,all_gen_prob.transpose(0,1),all_copy_prob.transpose(0,1)
		else:
			all_gen_op,all_gen_prob,all_copy_prob,add_ind,all_top_ind=self.create_batch_variable(target_length,doc_batch.size(1),mode,decoder_ip_batch.size(0))
			decoder_step_ip=decoder_ip_embed.clone()
			for step in range(target_length):
				gen_prob,copy_prob=self.gen_copy_prob(context_repr_original,doc_repr_original,hidden_state)
				output_scores,output_prob,hidden_state=self.decoder(decoder_step_ip,context_repr_original,doc_repr_original,hidden_state)
				comp_prob=copy_prob>gen_prob
				all_gen_op[step]=output_scores
				all_gen_prob[step]=gen_prob.squeeze(1)
				all_copy_prob[step]=copy_prob.squeeze(1)
				top_elem,top_ind=torch.topk(output_prob,1,dim=1)
				all_top_ind[step]=top_ind.squeeze()
				changed_top_ind=top_ind.clone()
				if self.prev_connection==True and step-1>=0:
					to_put,non_zero_ind=self.get_prev_connection(prev_comp_prob,doc_batch,end_max_ind,add_ind)
					if non_zero_ind.size()!=torch.LongTensor().size() and to_put.size()!=torch.LongTensor().size():
						changed_top_ind.data.index_copy_(0,non_zero_ind.data,to_put.data)
				decoder_step_ip=self.word_embed([changed_top_ind.squeeze(1)])[0]
				prev_comp_prob=comp_prob.clone().squeeze(1)
			return all_gen_op.transpose(0,1),start_attn_scores,end_attn_scores,start_max_ind,end_max_ind,all_gen_prob.transpose(0,1),all_copy_prob.transpose(0,1),all_top_ind.transpose(0,1)

	def get_prev_connection(self,step_comp_prob,doc_batch,end_max_ind,add_ind):

		non_zero_ind=step_comp_prob.nonzero()
		if non_zero_ind.size()!=torch.LongTensor().size():
			non_zero_ind=non_zero_ind.squeeze(1)            
			flat_end_max_ind=end_max_ind+add_ind
			end_max_word_id=doc_batch.view(-1,1).index_select(dim=0,index=flat_end_max_ind)
			to_put=end_max_word_id.index_select(dim=0,index=non_zero_ind)
			return to_put,non_zero_ind
		else:
			return torch.LongTensor(),torch.LongTensor()

	def passage_context_encode(self,query_embed,query,doc_embed,document,context_embed,context):

		query_pack=pack_padded_sequence(query_embed,query.sorted_lengths,batch_first=True)
		doc_pack=pack_padded_sequence(doc_embed,document.sorted_lengths,batch_first=True)
		context_pack=pack_padded_sequence(context_embed,context.sorted_lengths,batch_first=True)
		query_encoding,query_ht,doc_encoding,doc_ht,context_encoding,context_ht=self.sentence_enc(query_pack,doc_pack,context_pack)
		return query_encoding,query_ht,doc_encoding,doc_ht,context_encoding,context_ht

	def check_direction(self,query_repr,doc_ht):

		if self.no_directions>1:
			query_repr=torch.cat((query_repr[0],query_repr[1]),dim=1)
			doc_ht=torch.cat((doc_ht[0],doc_ht[1]),dim=1)
		else:
			query_repr=query_repr.squeeze(0)
			doc_ht=doc_ht.squeeze(0)
		return query_repr,doc_ht
		
	def create_batch_variable(self,res_seq_len,doc_seq_len,mode,batch_size):

		all_gen_op=Variable(torch.zeros(res_seq_len,batch_size,self.vocab_size))
		all_gen_prob=Variable(torch.zeros(res_seq_len,batch_size))
		all_copy_prob=Variable(torch.zeros(res_seq_len,batch_size))
		add_ind=Variable(torch.LongTensor([j*doc_seq_len for j in range(batch_size)]))
		if mode=='train':
			if torch.cuda.is_available():
				return all_gen_op.cuda(),all_gen_prob.cuda(),all_copy_prob.cuda(),add_ind.cuda()
			else:
				return all_gen_op,all_gen_prob,all_copy_prob,add_ind
		else:
			all_top_ind=Variable(torch.zeros(res_seq_len,batch_size)).long()
			if torch.cuda.is_available():
				return all_gen_op.cuda(),all_gen_prob.cuda(),all_copy_prob.cuda(),add_ind.cuda(),all_top_ind.cuda()
			else:	
				return all_gen_op,all_gen_prob,all_copy_prob,add_ind,all_top_ind