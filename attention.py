import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):

	def __init__(self,attn_size,hidden_size):
		
		super(Attention,self).__init__()
		self.attn=Gen_attention(attn_size,hidden_size)

	def forward(self,last_state,enc_outputs,attn_masks):

		attn_weights,scores=self.attn(last_state,enc_outputs,attn_masks)
		context=attn_weights.bmm(enc_outputs)
		return context.squeeze(1),attn_weights.squeeze(1),scores

class Gen_attention(nn.Module):

	def __init__(self,attn_size,hidden_size):

		super(Gen_attention,self).__init__()
		self.attn_mech=nn.Linear(attn_size,hidden_size)
		self.V=nn.Parameter(torch.randn(hidden_size))

	def forward(self,last_state,enc_outputs,attn_masks):
		
		seq_len=enc_outputs.size(1)
		hidden=last_state.repeat(seq_len,1,1).transpose(0,1)
		scores=self.unnorm_scores(hidden,enc_outputs)
		self.apply_masking(scores,attn_masks)
		norm_scores=F.softmax(scores,dim=1)
		return norm_scores.unsqueeze(1),scores
		
	def apply_masking(self,scores,attn_masks):
		
		#scores.data.masked_fill_(attn_masks.data!=1,-float('inf'))
		scores.data.masked_fill_(attn_masks.data!=1,-1e12)

	def unnorm_scores(self,last_state,enc_outputs):

		sim_param=F.tanh(self.attn_mech(torch.cat((last_state,enc_outputs),2)))
		sim_param=sim_param.transpose(2,1)
		v=self.V.repeat(enc_outputs.size(0),1).unsqueeze(1)
		score=torch.bmm(v,sim_param)
		return score.squeeze(1)