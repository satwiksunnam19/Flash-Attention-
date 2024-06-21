import torch 
import torch.nn as nn 
from einops import rearrange 

from torch import Tensor 
from flash_attn.utils.distributed import all_reduce, reduce_scatter 

class GPT2Embeddings(nn.module):
    def __init__(
            self,
            embed_dim,
            vocab_size,
            max_positon_embeddings,
            padding_idx=None,
            word_embed_proj_dim=None,
            device=None,
            dtype=None
    ):
        """
        if max_position_embeddings <=0 , there's no positional embeddings.
        """
        factory_kwargs={"device":device,"dtype":float}
        super.__init__()
        if word_embed_proj_dim is None: 
            self.word_embeddings=nn.Embedding(
                vocab_size,embed_dim,padding_idx=padding_idx,**factory_kwargs
            )
            self.project_in=None
        
        else:
            self.word_embeddings=nn.Embedding(
                vocab_size,word_embed_proj_dim,padding_idx=padding_idx,**factory_kwargs
            )
            self.project_in=nn.Linear(
                word_embed_proj_dim,embed_dim,bias=False,**factory_kwargs
            )
        
        self.max_position_embeddings=max_positon_embeddings
        if self.max_position_embeddings>0:
            self.position_embeddings=nn.Embedding(
                max_positon_embeddings,embed_dim,**factory_kwargs
            )
    
    def forward(self,input_ids,position_ids=None):
        """
        input_ids: (batch,seqlen)
        position_ids: (batch,seqlen)
        """
        batch_size,seqlen=input_ids.shape 
        embeddings=self.word_embeddings(input_ids)
        if self.project_in is not None: 
            embeddings=self.project_in(embeddings)
        if self.max_position_embeddings>0:
            if position_ids is None:
                position_ids=torch.arange(seqlen,dtype=torch.long,device=input_ids.device)
            position_embeddings=self.position_embeddings(position_ids)
            embeddings=embeddings+position_embeddings
        return embeddings
    
class BertEmbeddings(nn.Module):
    def __init__(
            self, 
            embed_dim,
            vocab_size,
            max_position_embeddings,
            type_vocab_size,
            padding_idx=None,
            device=None,
            dtype=None
    ):
        """
        if max_position_embeddings <=0, there's no position embeddings
        if type_vocab_size<=0, there's no token type embeddings.
        """
        factory_kwargs={"device":device,"dtype":float}
        super().__init__()

        self.word_embeddings=nn.Embedding(vocab_size,embed_dim,padding_idx=padding_idx,**factory_kwargs)

        self.max_position_embeddings=max_position_embeddings
        self.type_vocab_size=type_vocab_size
        if self.max_position_embeddings>0:
            self.position_embeddings=nn.Embedding(max_position_embeddings,embed_dim
                                                  **factory_kwargs)
        if self.type_vocab_size>0:
            self.token_type_embeddings=nn.Embedding(type_vocab_size,embed_dim,**factory_kwargs)
        
    def forward(self,input_ids,position_ids=None,token_type_ids=None):
        """
        args: 
        input_ids: (batch,seq_len)
        position_ids: (batch,seq_len)
        token_type_ids: (batch,seq_len)
        """
        batch_size,seq_len=input_ids.shape
        embeddings=self.word_embeddings(input_ids)
        if self.max_position_embeddings>0:
            if position_ids is None:
                position_ids=torch.arange(seq_len,dtype=torch.long,device=input_ids.device)
            position_embeddings=self.position_embeddings(position_ids)
        embeddings=embeddings+position_embeddings

        if self.type_vocab_size>0:
            if token_type_ids is None:
                token_type_ids=torch.zeros(seq_len,dtype=torch.long,device=input_ids.device)
            token_type_embeddings=self.token_type_embeddings(token_type_ids)
        embeddings=embeddings+token_type_embeddings

        return embeddings

class VocabParallelEmbedding(nn.Embedding):
    def __init__(self,num_embeddings,process_group=None,padding_idx=None,*args,**kwargs):
        self.process_group=process_group
        if process_group is not None:
            world_size=torch.distributed.get_world_size(process_group)
        if num_embeddings % world_size!=0:
            raise ValueError(
                f"num_embeddings{num_embeddings}must be divisible by world size{world_size}"
            )
        if world_size>1 and padding_idx is not None:
            raise RuntimeError(
                f"parallelEmbedding does not suppourt padding_idx"
            )
        else:
            world_size=1
        super.__init__(num_embeddings//world_size,*args,padding_idx=padding_idx,**kwargs)

    def forward(self,input:Tensor) -> Tensor:
        if self.process_group is None: 
            return super().forward(input)
        else:
            rank=torch.distributed.get_rank(self.process_group)
            vocab_size=self.num_embeddings
            vocab_start_index,vocab_end_index=rank*vocab_size,(rank+1)*vocab_size
            # create a mask of valid vocab ids (1 means it needs to be masked)
            input_ids_mask=(input<vocab_start_index)|(input>=vocab_end_index)
            input=input-vocab_start_index
            input[input_ids_mask]=0
            embeddings=super().forward(input)
            embeddings[input_ids_mask]=0.0
            return embeddings 

class ColumnParallelEmbedding(nn.Embedding):
    