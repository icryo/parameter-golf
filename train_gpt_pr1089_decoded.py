from __future__ import annotations
_y='passthrough_ctrl'
_x='mlp_down_bank'
_w='mlp_up_bank'
_v='kv_bank'
_u='qo_bank'
_t='.attn.'
_s='passthrough_orig_dtypes'
_r='dtypes'
_q='scales'
_p='quantized'
_o='per_row'
_n='scheme'
_m='tok_emb.weight'
_l='torch.'
_k='momentum'
_j='padded_grad'
_i='fineweb_train_*.bin'
_h='prune_room_bytes'
_g='headroom_kb'
_f='budget_mb'
_e='total_mb'
_d='promoted_mb'
_c='base_mb'
_b='attn'
_a='mlp'
_Z='.mlp.'
_Y='shard_mom'
_X='shard'
_W='row_col'
_V='9,10'
_U='.scale'
_T='.q'
_S='inf'
_R='passthrough'
_Q='cpu'
_P='scale'
_O='full_update'
_N='none'
_M='desc'
_L='lzma'
_K='brotli'
_J='cuda'
_I='utf-8'
_H='lr'
_G='params'
_F='1'
_E=.0
_D=False
_C=1.
_B=True
_A=None
import copy,glob,io,lzma,math,os,random,subprocess,sys,time,uuid,zlib
from pathlib import Path
import numpy as np,sentencepiece as spm,torch,torch.distributed as dist,torch.nn.functional as F
from torch import Tensor,nn
from torch.nn.parallel import DistributedDataParallel as DDP
try:import brotli as _brotli_probe;_COMPRESSOR=_K
except ImportError:_COMPRESSOR=_L
_BYTE_SHUFFLE=_B
_BYTE_SHUFFLE_STRIDE=2
_BSHF_MAGIC=b'BSHF'
def _byte_shuffle(data,stride=2):
	B=data;A=stride
	if A<=1 or len(B)<A:return B
	E=np.frombuffer(B,dtype=np.uint8);G=len(E);F=np.empty(G,dtype=np.uint8);C=0
	for H in range(A):D=E[H::A];F[C:C+len(D)]=D;C+=len(D)
	return _BSHF_MAGIC+bytes([A])+F.tobytes()
def _byte_unshuffle(data):
	A=data
	if len(A)<5 or A[:4]!=_BSHF_MAGIC:return A
	B=A[4]
	if B<2:return A[5:]
	F=np.frombuffer(A,dtype=np.uint8,offset=5);C=len(F);G=np.empty(C,dtype=np.uint8);D=0
	for H in range(B):E=C//B+(1 if H<C%B else 0);G[H::B][:E]=F[D:D+E];D+=E
	return G.tobytes()
class Hyperparameters:data_path=os.environ.get('DATA_PATH','./data/datasets/fineweb10B_sp1024');train_files=os.path.join(data_path,_i);val_files=os.path.join(data_path,'fineweb_val_*.bin');tokenizer_path=os.environ.get('TOKENIZER_PATH','./data/tokenizers/fineweb_1024_bpe.model');run_id=os.environ.get('RUN_ID',str(uuid.uuid4()));seed=int(os.environ.get('SEED',1337));val_batch_size=int(os.environ.get('VAL_BATCH_SIZE',524288));train_log_every=int(os.environ.get('TRAIN_LOG_EVERY',500));iterations=int(os.environ.get('ITERATIONS',20000));warmdown_iters=int(os.environ.get('WARMDOWN_ITERS',3500));lr_floor=float(os.environ.get('LR_FLOOR',.05));warmup_steps=int(os.environ.get('WARMUP_STEPS',20));train_batch_tokens=int(os.environ.get('TRAIN_BATCH_TOKENS',786432));train_seq_len=int(os.environ.get('TRAIN_SEQ_LEN',2048));eval_seq_len=int(os.environ.get('EVAL_SEQ_LEN',2048));max_wallclock_seconds=float(os.environ.get('MAX_WALLCLOCK_SECONDS',6e2));qk_gain_init=float(os.environ.get('QK_GAIN_INIT',1.5));vocab_size=int(os.environ.get('VOCAB_SIZE',1024));num_layers=int(os.environ.get('NUM_LAYERS',11));num_kv_heads=int(os.environ.get('NUM_KV_HEADS',4));model_dim=int(os.environ.get('MODEL_DIM',512));num_heads=int(os.environ.get('NUM_HEADS',8));mlp_mult=float(os.environ.get('MLP_MULT',3.5));tie_embeddings=bool(int(os.environ.get('TIE_EMBEDDINGS',_F)));rope_base=float(os.environ.get('ROPE_BASE',1e4));logit_softcap=float(os.environ.get('LOGIT_SOFTCAP',3e1));embed_lr=float(os.environ.get('EMBED_LR',.6));head_lr=float(os.environ.get('HEAD_LR',.008));tied_embed_lr=float(os.environ.get('TIED_EMBED_LR',.035));tied_embed_init_std=float(os.environ.get('TIED_EMBED_INIT_STD',.005));matrix_lr=float(os.environ.get('MATRIX_LR',.025));scalar_lr=float(os.environ.get('SCALAR_LR',.025));muon_momentum=float(os.environ.get('MUON_MOMENTUM',.99));muon_backend_steps=int(os.environ.get('MUON_BACKEND_STEPS',4));muon_momentum_warmup_start=float(os.environ.get('MUON_MOMENTUM_WARMUP_START',.92));muon_momentum_warmup_steps=int(os.environ.get('MUON_MOMENTUM_WARMUP_STEPS',1500));beta1=float(os.environ.get('BETA1',.9));beta2=float(os.environ.get('BETA2',.95));adam_eps=float(os.environ.get('ADAM_EPS',1e-08));grad_clip_norm=float(os.environ.get('GRAD_CLIP_NORM',.3));eval_stride=int(os.environ.get('EVAL_STRIDE',64));swa_enabled=bool(int(os.environ.get('SWA_ENABLED',_F)));swa_every=int(os.environ.get('SWA_EVERY',50));swa_threshold=float(os.environ.get('SWA_THRESHOLD',.2));muon_wd=float(os.environ.get('MUON_WD',.04));adam_wd=float(os.environ.get('ADAM_WD',.04));xsa_last_n=int(os.environ.get('XSA_LAST_N',11));rope_dims=int(os.environ.get('ROPE_DIMS',16));ln_scale=bool(int(os.environ.get('LN_SCALE',_F)));ve_enabled=bool(int(os.environ.get('VE_ENABLED',_F)));ve_dim=int(os.environ.get('VE_DIM',128));ve_layers=os.environ.get('VE_LAYERS',_V);ema_enabled=bool(int(os.environ.get('EMA_ENABLED',_F)));ema_decay=float(os.environ.get('EMA_DECAY',.997));embed_beta1=float(os.environ.get('EMBED_BETA1',.7));head_beta1=float(os.environ.get('HEAD_BETA1',.7));muon_post_norm=os.environ.get('MUON_POST_NORM',_W);qat_threshold=float(os.environ.get('QAT_THRESHOLD',.15));qat_clip_pct=float(os.environ.get('QAT_CLIP_PCT',.9995));late_qat=bool(int(os.environ.get('LATE_QAT',_F)));mixed_precision=bool(int(os.environ.get('MIXED_PRECISION',_F)));target_bytes_limit=int(os.environ.get('TARGET_BYTES',16000000));ngram_buckets=int(os.environ.get('NGRAM_BUCKETS',8192));ngram_heads=int(os.environ.get('NGRAM_HEADS',2));ngram_orders=int(os.environ.get('NGRAM_ORDERS',2));ngram_dim_per_head=int(os.environ.get('NGRAM_DIM_PER_HEAD',32));gptq_calib_batches=int(os.environ.get('GPTQ_CALIB_BATCHES',64));gptq_block_size=int(os.environ.get('GPTQ_BLOCK_SIZE',128));gptq_damp=float(os.environ.get('GPTQ_DAMP',.01));gptq_reserve_ms=float(os.environ.get('GPTQ_RESERVE_MS',14e3));gptq_col_order=os.environ.get('GPTQ_COL_ORDER',_M);gptq_single_pass=bool(int(os.environ.get('GPTQ_SINGLE_PASS',_F)));soft_round_qat=bool(int(os.environ.get('SOFT_ROUND_QAT',_F)));snapshot_post_hessian=bool(int(os.environ.get('SNAPSHOT_POST_HESSIAN','0')));load_snapshot=os.environ.get('LOAD_SNAPSHOT','')
_POLAR_COEFFS_FULL=[(8.28721201814563,-23.595886519098837,17.300387312530933),(4.107059111542203,-2.9478499167379106,.5448431082926601),(3.9486908534822946,-2.908902115962949,.5518191394370137),(3.3184196573706015,-2.488488024314874,.51004894012372),(2.300652019954817,-1.6689039845747493,.4188073119525673),(1.891301407787398,-1.2679958271945868,.37680408948524835),(1.875,-1.25,.375)]
_AOL_POLAR_COEFFS=_POLAR_COEFFS_FULL[1:]
def zeropower_via_newtonschulz5(G,steps=4,eps=1e-07):
	K=steps;A=G.bfloat16()
	if A.ndim==2:
		D=A.size(0)>A.size(1)
		if D:A=A.T
		B=A@A.T;C=_C/(B.abs().sum(dim=1).sqrt()+eps);A=C.unsqueeze(1)*A;B=C.unsqueeze(0)*B*C.unsqueeze(1)
		for E in range(K):
			F,H,I=_AOL_POLAR_COEFFS[min(E,len(_AOL_POLAR_COEFFS)-1)]
			if E>0:B=A@A.T
			J=H*B+I*B@B;A=F*A+J@A
		return A.T if D else A
	else:
		D=A.size(-2)>A.size(-1)
		if D:A=A.mT
		B=A@A.mT;C=_C/(B.abs().sum(dim=-1).sqrt()+eps);A=C.unsqueeze(-1)*A;B=C.unsqueeze(-2)*B*C.unsqueeze(-1)
		for E in range(K):
			F,H,I=_AOL_POLAR_COEFFS[min(E,len(_AOL_POLAR_COEFFS)-1)]
			if E>0:B=A@A.mT
			J=H*B+I*B@B;A=F*A+J@A
		return A.mT if D else A
def _post_ns_normalize(X,mode):
	A=mode
	if A==_N:return X
	if A in('row',_W):X=X/(X.float().norm(dim=-1,keepdim=_B).to(X.dtype)+1e-07)
	if A in('col',_W):X=X/(X.float().norm(dim=-2,keepdim=_B).to(X.dtype)+1e-07)
	return X
class Muon(torch.optim.Optimizer):
	def __init__(A,params,lr,momentum,backend_steps,nesterov=_B,weight_decay=_E,post_norm=_N):super().__init__(params,dict(lr=lr,momentum=momentum,backend_steps=backend_steps,nesterov=nesterov,weight_decay=weight_decay,post_norm=post_norm));A._built=_D
	def _build(A):
		A._distributed=dist.is_available()and dist.is_initialized();A._world_size=dist.get_world_size()if A._distributed else 1;A._rank=dist.get_rank()if A._distributed else 0;C=A._world_size;A._bank_meta=[]
		for I in A.param_groups:
			for B in I[_G]:G=B.shape[0];F=(G+C-1)//C*C;H=F//C;D=B.shape[1:];E=B.device;A._bank_meta.append({'p':B,'B':G,_j:torch.zeros(F,*D,device=E,dtype=torch.bfloat16),_X:torch.zeros(H,*D,device=E,dtype=torch.bfloat16),_Y:torch.zeros(H,*D,device=E,dtype=torch.bfloat16),_O:torch.zeros(F,*D,device=E,dtype=torch.bfloat16),_P:max(1,B.shape[-2]/B.shape[-1])**.5})
		A._bank_meta.sort(key=lambda m:-m['p'].numel());A._built=_B
	def launch_reduce_scatters(A):
		if not A._built:A._build()
		if not A._distributed:return
		A._rs_futures=[]
		for B in A._bank_meta:
			D=B['p']
			if D.grad is _A:A._rs_futures.append(_A);continue
			C=B[_j];C[:B['B']].copy_(D.grad.bfloat16())
			if C.shape[0]>B['B']:C[B['B']:].zero_()
			E=dist.reduce_scatter_tensor(B[_X],C,op=dist.ReduceOp.AVG,async_op=_B);A._rs_futures.append(E)
	@torch.no_grad()
	def step(self,closure=_A):
		U='_rs_futures';P=closure;O='momentum_buffer';A=self;Q=_A
		if P is not _A:
			with torch.enable_grad():Q=P()
		if not A._built:A._build()
		for E in A.param_groups:
			F=E[_H];R=E[_k];V=E['backend_steps'];W=E['nesterov'];G=E.get('weight_decay',_E);X=E.get('post_norm',_N);J=_A;B=_A;S=A._distributed and hasattr(A,U)
			for(T,H)in enumerate(A._bank_meta):
				I=H['p']
				if I.grad is _A:continue
				if J is not _A:
					J.wait();C=B['p'];M=B[_O][:B['B']];C.add_(M.to(dtype=C.dtype),alpha=-F*B[_P])
					if G>_E:C.data.mul_(_C-F*G)
				if S and A._rs_futures[T]is not _A:A._rs_futures[T].wait();K=H[_X];L=H[_Y]
				else:
					K=I.grad.bfloat16();N=A.state[I]
					if O not in N:N[O]=torch.zeros_like(K)
					L=N[O]
				L.mul_(R).add_(K)
				if W:D=K.add(L,alpha=R)
				else:D=L
				D=zeropower_via_newtonschulz5(D,steps=V);D=_post_ns_normalize(D,X)
				if S:J=dist.all_gather_into_tensor(H[_O],D,async_op=_B);B=H
				else:
					I.add_(D.to(dtype=I.dtype),alpha=-F*H[_P])
					if G>_E:I.data.mul_(_C-F*G)
			if J is not _A:
				J.wait();C=B['p'];M=B[_O][:B['B']];C.add_(M.to(dtype=C.dtype),alpha=-F*B[_P])
				if G>_E:C.data.mul_(_C-F*G)
			if hasattr(A,U):del A._rs_futures
		return Q
def build_sentencepiece_luts(sp,vocab_size,device):
	D=device;B=sp;G=int(B.vocab_size());E=max(G,vocab_size);F=np.zeros((E,),dtype=np.int16);H=np.zeros((E,),dtype=np.bool_);I=np.ones((E,),dtype=np.bool_)
	for A in range(G):
		if B.is_control(A)or B.is_unknown(A)or B.is_unused(A):continue
		I[A]=_D
		if B.is_byte(A):F[A]=1;continue
		C=B.id_to_piece(A)
		if C.startswith('▁'):H[A]=_B;C=C[1:]
		F[A]=len(C.encode(_I))
	return torch.tensor(F,dtype=torch.int16,device=D),torch.tensor(H,dtype=torch.bool,device=D),torch.tensor(I,dtype=torch.bool,device=D)
def load_validation_tokens(pattern,seq_len):
	B=pattern;A=seq_len;C=[Path(A)for A in sorted(glob.glob(B))]
	if not C:raise FileNotFoundError(f"No files found for pattern: {B}")
	D=torch.cat([load_data_shard(A)for A in C]).contiguous();E=(D.numel()-1)//A*A
	if E<=0:raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={A}")
	return D[:E+1]
def eval_val(args,model,rank,world_size,device,grad_accum_steps,val_tokens,base_bytes_lut,has_leading_space_lut,is_boundary_token_lut,eval_seq_len=_A):
	K=val_tokens;J=grad_accum_steps;F=model;E=args;C=device;B=world_size;A=eval_seq_len or E.train_seq_len;L=E.val_batch_size//(B*J)
	if L<A:raise ValueError(f"VAL_BATCH_SIZE must provide at least one sequence per rank; got VAL_BATCH_SIZE={E.val_batch_size}, WORLD_SIZE={B}, GRAD_ACCUM_STEPS={J}, seq_len={A}")
	M=L//A;N=(K.numel()-1)//A;W=N*rank//B;O=N*(rank+1)//B;G=torch.zeros((),device=C,dtype=torch.float64);D=torch.zeros((),device=C,dtype=torch.float64);H=torch.zeros((),device=C,dtype=torch.float64);F.eval()
	with torch.inference_mode():
		for P in range(W,O,M):
			X=min(P+M,O);Y=P*A;Z=X*A+1;Q=K[Y:Z].to(device=C,dtype=torch.int64,non_blocking=_B);R=Q[:-1].reshape(-1,A);I=Q[1:].reshape(-1,A)
			with torch.autocast(device_type=_J,dtype=torch.bfloat16,enabled=_B):a=F(R,I).detach()
			S=float(I.numel());G+=a.to(torch.float64)*S;D+=S;b=R.reshape(-1);T=I.reshape(-1);U=base_bytes_lut[T].to(dtype=torch.int16);U+=(has_leading_space_lut[T]&~is_boundary_token_lut[b]).to(dtype=torch.int16);H+=U.to(torch.float64).sum()
	if dist.is_available()and dist.is_initialized():dist.all_reduce(G,op=dist.ReduceOp.SUM);dist.all_reduce(D,op=dist.ReduceOp.SUM);dist.all_reduce(H,op=dist.ReduceOp.SUM)
	V=G/D;c=V.item()/math.log(2.);d=D.item()/H.item();F.train();return float(V.item()),float(c*d)
CONTROL_TENSOR_NAME_PATTERNS=tuple(A for A in os.environ.get('CONTROL_TENSOR_NAME_PATTERNS','attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,skip_gate,skip_gates,smear,ve_layer_scales,ve_shared.scale,ngram_gate').split(',')if A)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS=tuple(A for A in os.environ.get('INT8_KEEP_FLOAT_FP32_NAME_PATTERNS',','.join(CONTROL_TENSOR_NAME_PATTERNS)).split(',')if A)
INT8_KEEP_FLOAT_MAX_NUMEL=65536
INT8_KEEP_FLOAT_STORE_DTYPE=torch.float16
INT8_PER_ROW_SCALE_DTYPE=torch.float16
INT8_CLIP_PERCENTILE=99.99984
INT8_CLIP_Q=INT8_CLIP_PERCENTILE/1e2
def tensor_nbytes(t):return int(t.numel())*int(t.element_size())
def keep_float_tensor(name,t,passthrough_orig_dtypes):
	if any(A in name for A in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):return t.float().contiguous()
	if t.dtype in{torch.float32,torch.bfloat16}:passthrough_orig_dtypes[name]=str(t.dtype).removeprefix(_l);return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
	return t
def quantize_float_tensor(t):
	A=t.float()
	if A.ndim==2:B=torch.quantile(A.abs(),INT8_CLIP_Q,dim=1)if A.numel()else torch.empty((A.shape[0],),dtype=torch.float32);E=torch.maximum(torch.minimum(A,B[:,_A]),-B[:,_A]);C=(B/127.).clamp_min(_C/127.);D=torch.clamp(torch.round(E/C[:,_A]),-127,127).to(torch.int8).contiguous();return D,C.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
	B=float(torch.quantile(A.abs().flatten(),INT8_CLIP_Q).item())if A.numel()else _E;C=torch.tensor(B/127. if B>0 else _C,dtype=torch.float32);D=torch.clamp(torch.round(torch.clamp(A,-B,B)/C),-127,127).to(torch.int8).contiguous();return D,C
def quantize_state_dict_int8(state_dict):
	S='baseline_tensor_bytes';R='num_nonfloat_tensors';Q='num_float_tensors';P='num_tensors';O='param_count';D='int8_payload_bytes';J={};K={};L={};E={};F={};G={};A=dict.fromkeys((O,P,Q,R,S,D),0)
	for(C,T)in state_dict.items():
		B=T.detach().to(_Q).contiguous();A[O]+=int(B.numel());A[P]+=1;A[S]+=tensor_nbytes(B)
		if not B.is_floating_point():A[R]+=1;E[C]=B;A[D]+=tensor_nbytes(B);continue
		if B.numel()<=INT8_KEEP_FLOAT_MAX_NUMEL or C==_m:M=keep_float_tensor(C,B,F);E[C]=M;A[D]+=tensor_nbytes(M);continue
		A[Q]+=1;N,H=quantize_float_tensor(B)
		if H.ndim>0:G[C]={_n:_o,'axis':0}
		J[C]=N;K[C]=H;L[C]=str(B.dtype).removeprefix(_l);A[D]+=tensor_nbytes(N)+tensor_nbytes(H)
	I={'__quant_format__':'int8_clean_per_row_v1',_p:J,_q:K,_r:L,_R:E}
	if G:I['qmeta']=G
	if F:I[_s]=F
	return I,A
def dequantize_state_dict_int8(obj):
	B=obj;D={};I=B.get('qmeta',{});J=B.get(_s,{})
	for(A,E)in B[_p].items():
		G=getattr(torch,B[_r][A]);C=B[_q][A]
		if I.get(A,{}).get(_n)==_o or C.ndim>0:C=C.to(dtype=torch.float32);D[A]=(E.float()*C.view(E.shape[0],*[1]*(E.ndim-1))).to(dtype=G).contiguous()
		else:K=float(C.item());D[A]=(E.float()*K).to(dtype=G).contiguous()
	for(A,L)in B[_R].items():
		F=L.detach().to(_Q).contiguous();H=J.get(A)
		if isinstance(H,str):F=F.to(dtype=getattr(torch,H)).contiguous()
		D[A]=F
	return D
def load_data_shard(file):
	H='<u2';G='<i4';A=file;D=256*np.dtype(G).itemsize;I=np.dtype(H).itemsize;B=np.fromfile(A,dtype=G,count=256)
	if B.size!=256 or int(B[0])!=20240520 or int(B[1])!=1:raise ValueError(f"Unexpected shard header for {A}")
	C=int(B[2]);E=D+C*I
	if A.stat().st_size!=E:raise ValueError(f"Shard size mismatch for {A}: expected {E} bytes")
	F=np.fromfile(A,dtype=H,count=C,offset=D)
	if F.size!=C:raise ValueError(f"Short read for {A}")
	return torch.from_numpy(F.astype(np.uint16,copy=_D))
class TokenStream:
	def __init__(A,pattern):
		B=pattern;A.files=[Path(A)for A in sorted(glob.glob(B))]
		if not A.files:raise FileNotFoundError(f"No files found for pattern: {B}")
		A.file_idx=0;A.tokens=load_data_shard(A.files[0]);A.pos=0
	def _advance_file(A):A.file_idx=(A.file_idx+1)%len(A.files);A.tokens=load_data_shard(A.files[A.file_idx]);A.pos=0
	def take(A,n):
		B=[];C=n
		while C>0:
			E=A.tokens.numel()-A.pos
			if E<=0:A._advance_file();continue
			D=min(C,E);B.append(A.tokens[A.pos:A.pos+D]);A.pos+=D;C-=D
		return B[0]if len(B)==1 else torch.cat(B)
class DistributedTokenLoader:
	def __init__(A,pattern,rank,world_size,device):A.rank=rank;A.world_size=world_size;A.device=device;A.stream=TokenStream(pattern)
	def next_batch(A,global_tokens,seq_len,grad_accum_steps):C=seq_len;F=global_tokens//(A.world_size*grad_accum_steps);B=F+1;G=A.stream.take(B*A.world_size);D=A.rank*B;E=G[D:D+B].to(dtype=torch.int64);H=E[:-1].reshape(-1,C);I=E[1:].reshape(-1,C);return H.to(A.device,non_blocking=_B),I.to(A.device,non_blocking=_B)
class RMSNorm(nn.Module):
	def __init__(A,eps=_A):super().__init__();A.eps=eps
	def forward(A,x):return F.rms_norm(x,(x.size(-1),),eps=A.eps)
class CastedLinear(nn.Linear):
	_qat_enabled=_D;_qat_clip_pct=.9995;_qat_default_bits=5;_qat_soft_round=_D;_qat_soft_alpha=_A
	def forward(A,x):
		B=A.weight.to(x.dtype)
		if CastedLinear._qat_enabled and A.training and B.ndim==2:
			C=getattr(A,'_qat_bits',CastedLinear._qat_default_bits)
			if CastedLinear._qat_soft_round:B=_apply_qat_soft_round(B,A.weight,C,CastedLinear._qat_soft_alpha)
			else:B=_apply_qat_ste(B,A.weight,C)
		D=A.bias.to(x.dtype)if A.bias is not _A else _A;return F.linear(x,B,D)
def _apply_qat_ste(w_cast,w_fp32,bits):
	B=bits;A=w_cast
	if B<=0:return A
	C=(1<<B-1)-1;F=-(1<<B-1)
	with torch.no_grad():D=w_fp32.float();G=D.abs().amax(dim=-1)*CastedLinear._qat_clip_pct;E=(G/float(C)).clamp_min(_C/float(C));H=(torch.clamp(torch.round(D/E.unsqueeze(-1)),F,C)*E.unsqueeze(-1)).to(A.dtype)
	return A+(H-A).detach()
def _apply_qat_soft_round(w_cast,w_fp32,bits,alpha):
	D=w_fp32;C=w_cast;A=bits
	if A<=0:return C
	H=C.dtype;B=(1<<A-1)-1;I=-(1<<A-1);J=D.float().detach().abs().amax(dim=-1)*CastedLinear._qat_clip_pct;E=(J/float(B)).clamp_min(_C/float(B));K=D.float()/E.unsqueeze(-1);F=K.clamp(float(I),float(B));G=F.detach().floor();L=F-G;M=G+torch.sigmoid(alpha*(L-.5));N=M*E.unsqueeze(-1);return N.to(H)
def _apply_bank_qat(w,bits,dtype):
	A=w.to(dtype)
	if CastedLinear._qat_enabled and torch.is_grad_enabled()and w.ndim==2:
		if CastedLinear._qat_soft_round:return _apply_qat_soft_round(A,w,bits,CastedLinear._qat_soft_alpha)
		return _apply_qat_ste(A,w,bits)
	return A
def restore_low_dim_params_to_fp32(module):
	with torch.no_grad():
		for(B,A)in module.named_parameters():
			if(A.ndim<2 or any(A in B for A in CONTROL_TENSOR_NAME_PATTERNS))and A.dtype!=torch.float32:A.data=A.data.float()
class Rotary(nn.Module):
	def __init__(A,dim,base=1e4,train_seq_len=1024,rope_dims=0):B=rope_dims;super().__init__();A.dim=dim;A.base=base;A.train_seq_len=train_seq_len;A.rope_dims=B if B>0 else dim;C=_C/base**(torch.arange(0,A.rope_dims,2,dtype=torch.float32)/A.rope_dims);A.register_buffer('inv_freq',C,persistent=_D);A._seq_len_cached=0;A._cos_cached=_A;A._sin_cached=_A
	def forward(A,seq_len,device,dtype):
		F=dtype;C=device;B=seq_len
		if A._cos_cached is _A or A._sin_cached is _A or A._seq_len_cached!=B or A._cos_cached.device!=C:
			D=A.rope_dims
			if B>A.train_seq_len:H=B/A.train_seq_len;I=A.base*H**(D/(D-2));E=_C/I**(torch.arange(0,D,2,dtype=torch.float32,device=C)/D)
			else:E=A.inv_freq.to(C)
			J=torch.arange(B,device=C,dtype=E.dtype);G=torch.outer(J,E);A._cos_cached=G.cos()[_A,:,_A,:];A._sin_cached=G.sin()[_A,:,_A,:];A._seq_len_cached=B
		return A._cos_cached.to(dtype=F),A._sin_cached.to(dtype=F)
def apply_rotary_emb(x,cos,sin,rope_dims=0):
	F=sin;E=cos;A=rope_dims
	if A>0 and A<x.size(-1):G,H=x[...,:A],x[...,A:];B=A//2;C,D=G[...,:B],G[...,B:];G=torch.cat((C*E+D*F,C*-F+D*E),dim=-1);return torch.cat((G,H),dim=-1)
	B=x.size(-1)//2;C,D=x[...,:B],x[...,B:];return torch.cat((C*E+D*F,C*-F+D*E),dim=-1)
class CausalSelfAttention(nn.Module):
	def __init__(A,dim,num_heads,num_kv_heads,rope_base,qk_gain_init):
		C=num_kv_heads;B=num_heads;super().__init__()
		if dim%B!=0:raise ValueError('model_dim must be divisible by num_heads')
		if B%C!=0:raise ValueError('num_heads must be divisible by num_kv_heads')
		A.num_heads=B;A.num_kv_heads=C;A.head_dim=dim//B
		if A.head_dim%2!=0:raise ValueError('head_dim must be even for RoPE')
		A.q_gain=nn.Parameter(torch.full((B,),qk_gain_init,dtype=torch.float32));A.rope_dims=0;A.rotary=Rotary(A.head_dim,base=rope_base,train_seq_len=1024);A.use_xsa=_D
	def _xsa_efficient(K,y,v):A,B,C,D=y.shape;E=v.size(-2);I=C//E;G=y.reshape(A,B,E,I,D);H=F.normalize(v,dim=-1).unsqueeze(-2);J=(G*H).sum(dim=-1,keepdim=_B)*H;return(G-J).reshape(A,B,C,D)
	def forward(A,x,q_w,k_w,v_w,out_w,v_embed=_A):
		J=v_embed;H,G,M=x.shape;I=CastedLinear._qat_default_bits;B=F.linear(x,_apply_bank_qat(q_w,I,x.dtype)).reshape(H,G,A.num_heads,A.head_dim);C=F.linear(x,_apply_bank_qat(k_w,I,x.dtype)).reshape(H,G,A.num_kv_heads,A.head_dim);D=F.linear(x,_apply_bank_qat(v_w,I,x.dtype))
		if J is not _A:D=D+J
		D=D.reshape(H,G,A.num_kv_heads,A.head_dim);B=F.rms_norm(B,(B.size(-1),));C=F.rms_norm(C,(C.size(-1),));K,L=A.rotary(G,x.device,B.dtype);B=apply_rotary_emb(B,K,L,A.rope_dims);C=apply_rotary_emb(C,K,L,A.rope_dims);B=B*A.q_gain.to(dtype=B.dtype)[_A,_A,:,_A];E=F.scaled_dot_product_attention(B.transpose(1,2),C.transpose(1,2),D.transpose(1,2),is_causal=_B,enable_gqa=A.num_kv_heads!=A.num_heads).transpose(1,2)
		if A.use_xsa:E=A._xsa_efficient(E,D)
		E=E.reshape(H,G,M);return F.linear(E,_apply_bank_qat(out_w,I,E.dtype)),_A
class SmearGate(nn.Module):
	def __init__(A,dim):super().__init__();A.gate=nn.Parameter(torch.zeros(dim,dtype=torch.float32))
	def forward(A,x):B=torch.sigmoid(A.gate.to(dtype=x.dtype))[_A,_A,:];C=F.pad(x[:,:-1],(0,0,1,0));return torch.lerp(x,C,B)
class EngramLite(nn.Module):
	def __init__(A,num_buckets,num_heads,num_orders,dim_per_head,model_dim):F=model_dim;E=num_buckets;D=dim_per_head;C=num_orders;B=num_heads;super().__init__();A.num_buckets=E;A.num_heads=B;A.num_orders=C;A.dim_per_head=D;G=C*B*E;H=C*B*D;A.embed=nn.Embedding(G,D);nn.init.normal_(A.embed.weight,std=.01);A.proj=CastedLinear(H,F,bias=_D);A.proj._zero_init=_B;A.ngram_gate=nn.Parameter(torch.zeros(F,dtype=torch.float32))
	def forward(C,input_ids):
		B=input_ids;A=C.num_buckets;D=F.pad(B[:,:-1],(1,0),value=0);J=(D*1009+B)%A;K=(D*2719+314159^B*3137)%A;E=[J,K+A]
		if C.num_orders>=2:G=F.pad(D[:,:-1],(1,0),value=0);L=(G*36313^D*27191^B*4903)%A;M=(G*7919^D*4391^B*6151)%A;H=2*A;E.extend([L+H,M+H+A])
		N=torch.stack(E,dim=-1);O=C.embed(N);P=O.reshape(*B.shape,-1);I=C.proj(P);Q=torch.sigmoid(C.ngram_gate.to(dtype=I.dtype))[_A,_A,:];return I*Q
class ValueEmbedding(nn.Module):
	def __init__(A,vocab_size,ve_dim,kv_dim):
		C=kv_dim;B=ve_dim;super().__init__();A.embed=nn.Embedding(vocab_size,B);nn.init.normal_(A.embed.weight,std=.01);A.proj=CastedLinear(B,C,bias=_D)if B!=C else _A
		if A.proj is not _A:nn.init.zeros_(A.proj.weight)
		A.scale=nn.Parameter(torch.tensor(.1,dtype=torch.float32))
	def forward(A,token_ids):
		B=A.embed(token_ids)
		if A.proj is not _A:B=A.proj(B)
		return B*A.scale.to(dtype=B.dtype)
class MLP(nn.Module):
	def __init__(A,dim,mlp_mult):super().__init__()
	def forward(B,x,up_w,down_w):A=CastedLinear._qat_default_bits;x=F.leaky_relu(F.linear(x,_apply_bank_qat(up_w,A,x.dtype)),negative_slope=.3);return F.linear(x.square(),_apply_bank_qat(down_w,A,x.dtype))
class Block(nn.Module):
	def __init__(A,dim,num_heads,num_kv_heads,mlp_mult,rope_base,qk_gain_init,layer_idx=0,ln_scale=_D):B=dim;super().__init__();A.attn_norm=RMSNorm();A.mlp_norm=RMSNorm();A.attn=CausalSelfAttention(B,num_heads,num_kv_heads,rope_base,qk_gain_init);A.mlp=MLP(B,mlp_mult);A.attn_scale=nn.Parameter(torch.ones(B,dtype=torch.float32));A.mlp_scale=nn.Parameter(torch.ones(B,dtype=torch.float32));A.resid_mix=nn.Parameter(torch.stack((torch.ones(B),torch.zeros(B))).float());A.ln_scale_factor=_C/math.sqrt(layer_idx+1)if ln_scale else _C
	def forward(A,x,x0,q_w,k_w,v_w,out_w,up_w,down_w,v_embed=_A):D=A.resid_mix.to(dtype=x.dtype);C=D[0][_A,_A,:]*x+D[1][_A,_A,:]*x0;E,F=A.attn(A.attn_norm(C)*A.ln_scale_factor,q_w,k_w,v_w,out_w,v_embed=v_embed);B=C+A.attn_scale.to(dtype=C.dtype)[_A,_A,:]*E;B=B+A.mlp_scale.to(dtype=B.dtype)[_A,_A,:]*A.mlp(A.mlp_norm(B)*A.ln_scale_factor,up_w,down_w);return B,F
class GPT(nn.Module):
	def __init__(A,vocab_size,num_layers,model_dim,num_heads,num_kv_heads,mlp_mult,tie_embeddings,tied_embed_init_std,logit_softcap,rope_base,qk_gain_init,ngram_buckets=0,ngram_heads=2,ngram_orders=2,ngram_dim_per_head=32,xsa_last_n=0,rope_dims=0,ln_scale=_D,ve_enabled=_D,ve_dim=128,ve_layers=_V):
		N=xsa_last_n;M=ngram_buckets;L=rope_base;K=tie_embeddings;J=mlp_mult;H=rope_dims;G=logit_softcap;F=num_kv_heads;E=vocab_size;D=num_heads;C=num_layers;B=model_dim;super().__init__();A._ve_target_dim=F*(B//D)
		if G<=_E:raise ValueError(f"logit_softcap must be positive, got {G}")
		A.tie_embeddings=K;A.tied_embed_init_std=tied_embed_init_std;A.logit_softcap=G;A.tok_emb=nn.Embedding(E,B);A.bigram=EngramLite(M,ngram_heads,ngram_orders,ngram_dim_per_head,B)if M>0 else _A;A.smear=SmearGate(B);A.num_encoder_layers=C//2;A.num_decoder_layers=C-A.num_encoder_layers;A.num_skip_weights=min(A.num_encoder_layers,A.num_decoder_layers);A.skip_weights=nn.Parameter(torch.ones(A.num_skip_weights,B,dtype=torch.float32));A.skip_gates=nn.Parameter(torch.zeros(A.num_skip_weights,B,dtype=torch.float32));I=B//D;Q=F*I;O=int(J*B);A.num_layers=C;A.qo_bank=nn.Parameter(torch.empty(2*C,B,B));A.kv_bank=nn.Parameter(torch.empty(2*C,Q,B));A.mlp_up_bank=nn.Parameter(torch.empty(C,O,B));A.mlp_down_bank=nn.Parameter(torch.empty(C,B,O));A.blocks=nn.ModuleList([Block(B,D,F,J,L,qk_gain_init,layer_idx=A,ln_scale=ln_scale)for A in range(C)])
		if H>0:
			I=B//D
			for P in A.blocks:P.attn.rope_dims=H;P.attn.rotary=Rotary(I,base=L,train_seq_len=1024,rope_dims=H)
		A.ve_layer_indices=[int(A)for A in ve_layers.split(',')if A.strip()]if ve_enabled else[];R=A._ve_target_dim
		if A.ve_layer_indices:A.ve_shared=ValueEmbedding(E,ve_dim,R);A.ve_layer_scales=nn.ParameterList([nn.Parameter(torch.ones(1,dtype=torch.float32))for A in A.ve_layer_indices])
		else:A.ve_shared=_A;A.ve_layer_scales=nn.ParameterList()
		A.value_embeds=nn.ModuleList();A.final_norm=RMSNorm();A.lm_head=_A if K else CastedLinear(B,E,bias=_D)
		if A.lm_head is not _A:A.lm_head._zero_init=_B
		if N>0:
			for S in range(max(0,C-N),C):A.blocks[S].attn.use_xsa=_B
		A._init_weights()
	def _init_weights(A):
		if A.tie_embeddings:nn.init.normal_(A.tok_emb.weight,mean=_E,std=A.tied_embed_init_std)
		D=A.num_layers;K=.05;E=A.qo_bank.shape[1]//A.blocks[0].attn.num_heads;G=A.blocks[0].attn.num_kv_heads;L=A.blocks[0].attn.num_heads;H=L//G
		for B in range(D):
			nn.init.orthogonal_(A.qo_bank.data[B],gain=_C);nn.init.orthogonal_(A.kv_bank.data[B],gain=_C);nn.init.orthogonal_(A.kv_bank.data[D+B],gain=_C);nn.init.orthogonal_(A.mlp_up_bank.data[B],gain=_C);nn.init.zeros_(A.mlp_down_bank.data[B]);M=A.kv_bank.data[D+B];I=torch.zeros_like(A.qo_bank.data[D+B])
			for F in range(G):
				N=M[F*E:(F+1)*E,:]
				for O in range(H):J=F*H+O;I[J*E:(J+1)*E,:]=-K*N
			A.qo_bank.data[D+B].copy_(I)
		for(P,C)in A.named_modules():
			if isinstance(C,nn.Linear):
				if getattr(C,'_zero_init',_D):nn.init.zeros_(C.weight)
				elif C.weight.ndim==2 and C.weight.shape[0]>=64 and C.weight.shape[1]>=64:nn.init.orthogonal_(C.weight,gain=_C)
	def _compute_ve_base(A,input_ids):
		if A.ve_shared is _A:return
		return A.ve_shared(input_ids)
	def _get_ve(A,layer_idx,ve_base):
		C=layer_idx;B=ve_base
		if B is _A or C not in A.ve_layer_indices:return
		D=A.ve_layer_indices.index(C);return B*A.ve_layer_scales[D].to(dtype=B.dtype)
	def forward(A,input_ids,target_ids):
		G=input_ids;E=A.num_layers;B=A.tok_emb(G)
		if A.bigram is not _A:B=B+A.bigram(G)
		B=F.rms_norm(B,(B.size(-1),));B=A.smear(B);J=B;H=[];K=A._compute_ve_base(G)
		for C in range(A.num_encoder_layers):I=A._get_ve(C,K);B,N=A.blocks[C](B,J,A.qo_bank[C],A.kv_bank[C],A.kv_bank[E+C],A.qo_bank[E+C],A.mlp_up_bank[C],A.mlp_down_bank[C],v_embed=I);H.append(B)
		for C in range(A.num_decoder_layers):
			D=A.num_encoder_layers+C
			if H:O=H.pop();P=torch.sigmoid(A.skip_gates[C].to(dtype=B.dtype))[_A,_A,:];Q=A.skip_weights[C].to(dtype=B.dtype)[_A,_A,:]*O;B=torch.lerp(Q,B,P)
			I=A._get_ve(D,K);B,N=A.blocks[D](B,J,A.qo_bank[D],A.kv_bank[D],A.kv_bank[E+D],A.qo_bank[E+D],A.mlp_up_bank[D],A.mlp_down_bank[D],v_embed=I)
		B=A.final_norm(B);L=B.reshape(-1,B.size(-1));R=target_ids.reshape(-1)
		if A.tie_embeddings:M=F.linear(L,A.tok_emb.weight)
		else:
			if A.lm_head is _A:raise RuntimeError('lm_head is required when tie_embeddings=False')
			M=A.lm_head(L)
		S=A.logit_softcap*torch.tanh(M/A.logit_softcap);return F.cross_entropy(S.float(),R,reduction='mean')
	def forward_logits(A,input_ids):
		G=input_ids;E=A.num_layers;B=A.tok_emb(G)
		if A.bigram is not _A:B=B+A.bigram(G)
		B=F.rms_norm(B,(B.size(-1),));B=A.smear(B);J=B;H=[];K=A._compute_ve_base(G)
		for C in range(A.num_encoder_layers):I=A._get_ve(C,K);B,M=A.blocks[C](B,J,A.qo_bank[C],A.kv_bank[C],A.kv_bank[E+C],A.qo_bank[E+C],A.mlp_up_bank[C],A.mlp_down_bank[C],v_embed=I);H.append(B)
		for C in range(A.num_decoder_layers):
			D=A.num_encoder_layers+C
			if H:N=H.pop();O=torch.sigmoid(A.skip_gates[C].to(dtype=B.dtype))[_A,_A,:];P=A.skip_weights[C].to(dtype=B.dtype)[_A,_A,:]*N;B=torch.lerp(P,B,O)
			I=A._get_ve(D,K);B,M=A.blocks[D](B,J,A.qo_bank[D],A.kv_bank[D],A.kv_bank[E+D],A.qo_bank[E+D],A.mlp_up_bank[D],A.mlp_down_bank[D],v_embed=I)
		B=A.final_norm(B)
		if A.tie_embeddings:L=F.linear(B,A.tok_emb.weight)
		else:L=A.lm_head(B)
		return A.logit_softcap*torch.tanh(L/A.logit_softcap)
def eval_val_sliding(args,base_model,rank,world_size,device,val_tokens,base_bytes_lut,has_leading_space_lut,is_boundary_token_lut,stride,batch_seqs=32,eval_seq_len=_A):
	T=batch_seqs;S=stride;R=val_tokens;Q=world_size;I=base_model;C=device;D=eval_seq_len or args.train_seq_len;J=R.numel()-1;U=[A for A in range(0,J,S)if min(A+D,J)-A>=1];V=len(U);f=V*rank//Q;g=V*(rank+1)//Q;W=U[f:g];K=torch.zeros((),device=C,dtype=torch.float64);G=torch.zeros((),device=C,dtype=torch.float64);L=torch.zeros((),device=C,dtype=torch.float64);I.eval();h=torch.compile(I.forward_logits,dynamic=_B,fullgraph=_B)
	with torch.inference_mode():
		for X in range(0,len(W),T):
			M=W[X:X+T];N=len(M);O=torch.zeros(N,D,dtype=torch.int64,device=C);P=torch.zeros(N,D,dtype=torch.int64,device=C);Y=[]
			for(B,E)in enumerate(M):Z=min(E+D,J);A=Z-E;Y.append(A);a=R[E:Z+1].to(dtype=torch.int64,device=C);O[B,:A]=a[:-1];P[B,:A]=a[1:]
			with torch.autocast(device_type=_J,dtype=torch.bfloat16):b=h(O)
			i=F.cross_entropy(b.reshape(-1,b.size(-1)).float(),P.reshape(-1),reduction=_N).reshape(N,D)
			for(B,E)in enumerate(M):A=Y[B];H=0 if E==0 else max(A-S,0);j=i[B,H:A].to(torch.float64);K+=j.sum();G+=float(A-H);c=P[B,H:A];k=O[B,H:A];d=base_bytes_lut[c].to(torch.float64);d+=(has_leading_space_lut[c]&~is_boundary_token_lut[k]).to(torch.float64);L+=d.sum()
	if dist.is_available()and dist.is_initialized():dist.all_reduce(K,op=dist.ReduceOp.SUM);dist.all_reduce(G,op=dist.ReduceOp.SUM);dist.all_reduce(L,op=dist.ReduceOp.SUM)
	e=(K/G).item();l=e/math.log(2.);m=G.item()/L.item();I.train();return e,l*m
def _classify_param(name):
	A=name
	if'tok_emb'in A or'lm_head'in A:return'embed'
	if _Z in A:return _a
	if _t in A or'.proj.'in A and _Z not in A:return _b
	return'other'
def quantize_int6_per_row(t,clip_range=31):
	A=clip_range;B=t.float()
	if B.ndim==2:
		E,F,G=_A,_A,float(_S)
		for H in[.999,.9995,.9999,.99999,_C]:
			if H<_C:I=torch.quantile(B.abs(),H,dim=1)
			else:I=B.abs().amax(dim=1)
			D=(I/A).clamp_min(_C/A).to(torch.float16);C=torch.clamp(torch.round(B/D.float()[:,_A]),-A,A).to(torch.int8);M=C.float()*D.float()[:,_A];J=(B-M).pow(2).mean().item()
			if J<G:E,F,G=C,D,J
		return E,F
	K=B.abs().max().item();L=torch.tensor(K/A if K>0 else _C,dtype=torch.float16);C=torch.clamp(torch.round(B/L.float()),-A,A).to(torch.int8);return C,L
def _precompute_row_scales(W,qmax):
	B=qmax;A=W.float();C=A.abs().amax(dim=1)/float(B);C=C.clamp_min(_C/float(B));F=torch.full((A.shape[0],),float(_S),device=A.device)
	for G in[.999,.9995,.9999,.99999,_C]:
		if G<_C:H=torch.quantile(A.abs(),G,dim=1)
		else:H=A.abs().amax(dim=1)
		E=(H/float(B)).clamp_min(_C/float(B));J=torch.clamp(torch.round(A/E[:,_A]),-B,B);K=J*E[:,_A];I=(A-K).pow(2).mean(dim=1);D=I<F;C[D]=E[D];F[D]=I[D]
	return C
def _gptq_block_sweep(W,Hinv,sf,qmin,qmax,block_size):
	F=block_size;G,D=W.shape;H=torch.zeros_like(W,dtype=torch.int8);I=W.clone()
	for C in range(0,D,F):
		A=min(C+F,D);E=A-C;J=I[:,C:A].clone();K=torch.zeros(G,E,dtype=torch.int8,device=W.device);L=torch.zeros(G,E,device=W.device);M=Hinv[C:A,C:A]
		for B in range(E):N=J[:,B];Q=M[B,B];O=torch.clamp(torch.round(N/sf),qmin,qmax).to(torch.int8);K[:,B]=O;P=(N-O.float()*sf)/Q;J[:,B:]-=P.unsqueeze(1)*M[B,B:].unsqueeze(0);L[:,B]=P
		H[:,C:A]=K
		if A<D:I[:,A:]-=L@Hinv[C:A,A:]
	return H
def quantize_int6_gptq(weight,hessian=_A,clip_range=31,block_size=128,damp_factor=.01,col_order=_M,single_pass=_D):
	O=damp_factor;N=block_size;M=weight;I=hessian;A=clip_range;Z=I.device if I is not _A else M.device;C=M.float().to(Z)
	if C.ndim!=2 or I is _A:return _quantize_int6_percentile(C,A)
	c,a=C.shape;B=I.float().clone();L=torch.diag(B);D=L==0;P=O*(torch.mean(L[~D])if not D.all()else torch.tensor(_C,device=B.device))if D.any()else O*torch.mean(L);Q=torch.arange(a,device=B.device);B[Q,Q]+=P;B[D,D]=P;G=torch.argsort(torch.diag(B),descending=col_order==_M);R=torch.argsort(G);H=C[:,G].clone();H[:,D[G]]=0;B=B[G][:,G]
	try:E=torch.linalg.cholesky(B);E=torch.cholesky_inverse(E);E=torch.linalg.cholesky(E,upper=_B)
	except torch.linalg.LinAlgError:return _quantize_int6_percentile(C,A)
	if single_pass:S=_precompute_row_scales(H,A);J=S.float();F=_gptq_block_sweep(H,E,J,-A,A,N);F=F[:,R];return F.cpu(),S.to(torch.float16).cpu()
	K=_A;T=_A;U=float(_S)
	for V in[.999,.9995,.9999,.99999,_C]:
		if V<_C:W=torch.quantile(C.abs(),V,dim=1)
		else:W=C.abs().amax(dim=1)
		X=(W/A).clamp_min(_C/A).to(torch.float16);J=X.float();F=_gptq_block_sweep(H,E,J,-A,A,N);b=F.float()*J[:,_A];Y=(H-b).pow(2).mean().item()
		if Y<U:K,T,U=F,X,Y
	K=K[:,R];return K.cpu(),T.cpu()
def _quantize_int6_percentile(t32,clip_range=31):
	B=clip_range;A=t32
	if A.ndim==2:
		E,F,G=_A,_A,float(_S)
		for H in[.999,.9995,.9999,.99999,_C]:
			if H<_C:I=torch.quantile(A.abs(),H,dim=1)
			else:I=A.abs().amax(dim=1)
			D=(I/B).clamp_min(_C/B).to(torch.float16);C=torch.clamp(torch.round(A/D.float()[:,_A]),-B,B).to(torch.int8);M=C.float()*D.float()[:,_A];J=(A-M).pow(2).mean().item()
			if J<G:E,F,G=C,D,J
		return E,F
	K=A.abs().max().item();L=torch.tensor(K/B if K>0 else _C,dtype=torch.float16);C=torch.clamp(torch.round(A/L.float()),-B,B).to(torch.int8);return C,L
def _unbank_state_dict(sd,num_layers):
	B={};D=num_layers
	for(E,C)in sd.items():
		if E==_u:
			for A in range(D):B[f"blocks.{A}.attn.c_q.weight"]=C[A];B[f"blocks.{A}.attn.proj.weight"]=C[D+A]
		elif E==_v:
			for A in range(D):B[f"blocks.{A}.attn.c_k.weight"]=C[A];B[f"blocks.{A}.attn.c_v.weight"]=C[D+A]
		elif E==_w:
			for A in range(D):B[f"blocks.{A}.mlp.fc.weight"]=C[A]
		elif E==_x:
			for A in range(D):B[f"blocks.{A}.mlp.proj.weight"]=C[A]
		else:B[E]=C
	return B
def _rebank_state_dict(sd,num_layers,template_sd=_A):
	C={};A=num_layers;G=set();D=[_A]*(2*A);E=[_A]*(2*A);H=[_A]*A;I=[_A]*A;M={'attn.c_q.weight':(D,0),'attn.proj.weight':(D,A),'attn.c_k.weight':(E,0),'attn.c_v.weight':(E,A),'mlp.fc.weight':(H,0),'mlp.proj.weight':(I,0)}
	for J in range(A):
		for(N,(O,P))in M.items():
			B=f"blocks.{J}.{N}"
			if B in sd:O[P+J]=sd[B];G.add(B)
	for(K,F)in[(_u,D),(_v,E),(_w,H),(_x,I)]:
		if not any(A is not _A for A in F):continue
		L=[A for(A,B)in enumerate(F)if B is _A]
		if L:raise ValueError(f"_rebank_state_dict: {K} missing slice indices {L}")
		C[K]=torch.stack(F)
	for(B,Q)in sd.items():
		if B not in G:C[B]=Q
	return C
class _HessianAttn(nn.Module):
	def __init__(A,dim,num_heads,num_kv_heads,rope_base,qk_gain_init):D=num_kv_heads;C=num_heads;B=dim;super().__init__();A.num_heads,A.num_kv_heads=C,D;A.head_dim=B//C;E=D*A.head_dim;A.c_q=CastedLinear(B,B,bias=_D);A.c_k=CastedLinear(B,E,bias=_D);A.c_v=CastedLinear(B,E,bias=_D);A.proj=CastedLinear(B,B,bias=_D);A.q_gain=nn.Parameter(torch.full((C,),qk_gain_init,dtype=torch.float32));A.rope_dims=0;A.rotary=Rotary(A.head_dim,base=rope_base,train_seq_len=1024);A.use_xsa=_D
	def _xsa_efficient(K,y,v):A,B,C,D=y.shape;E=v.size(-2);I=C//E;G=y.reshape(A,B,E,I,D);H=F.normalize(v,dim=-1).unsqueeze(-2);J=(G*H).sum(dim=-1,keepdim=_B)*H;return(G-J).reshape(A,B,C,D)
	def forward(A,x,v_embed=_A):
		I=v_embed;G,E,L=x.shape;B=A.c_q(x).reshape(G,E,A.num_heads,A.head_dim);C=A.c_k(x).reshape(G,E,A.num_kv_heads,A.head_dim);D=A.c_v(x)
		if I is not _A:D=D+I
		D=D.reshape(G,E,A.num_kv_heads,A.head_dim);B=F.rms_norm(B,(B.size(-1),));C=F.rms_norm(C,(C.size(-1),));J,K=A.rotary(E,x.device,B.dtype);B=apply_rotary_emb(B,J,K,A.rope_dims);C=apply_rotary_emb(C,J,K,A.rope_dims);B=B*A.q_gain.to(dtype=B.dtype)[_A,_A,:,_A];H=F.scaled_dot_product_attention(B.transpose(1,2),C.transpose(1,2),D.transpose(1,2),is_causal=_B,enable_gqa=A.num_kv_heads!=A.num_heads).transpose(1,2)
		if A.use_xsa:H=A._xsa_efficient(H,D)
		return A.proj(H.reshape(G,E,L))
class _HessianMLP(nn.Module):
	def __init__(B,dim,mlp_mult):C=mlp_mult;A=dim;super().__init__();B.fc=CastedLinear(A,int(C*A),bias=_D);B.proj=CastedLinear(int(C*A),A,bias=_D)
	def forward(A,x):return A.proj(F.leaky_relu(A.fc(x),negative_slope=.3).square())
class _HessianBlock(nn.Module):
	def __init__(A,dim,num_heads,num_kv_heads,mlp_mult,rope_base,qk_gain_init,layer_idx=0,ln_scale=_D):B=dim;super().__init__();A.attn_norm=RMSNorm();A.mlp_norm=RMSNorm();A.attn=_HessianAttn(B,num_heads,num_kv_heads,rope_base,qk_gain_init);A.mlp=_HessianMLP(B,mlp_mult);A.attn_scale=nn.Parameter(torch.ones(B,dtype=torch.float32));A.mlp_scale=nn.Parameter(torch.ones(B,dtype=torch.float32));A.resid_mix=nn.Parameter(torch.stack((torch.ones(B),torch.zeros(B))).float());A.ln_scale_factor=_C/math.sqrt(layer_idx+1)if ln_scale else _C
	def forward(A,x,x0,v_embed=_A):D=A.resid_mix.to(dtype=x.dtype);C=D[0][_A,_A,:]*x+D[1][_A,_A,:]*x0;E=A.attn(A.attn_norm(C)*A.ln_scale_factor,v_embed=v_embed);B=C+A.attn_scale.to(dtype=C.dtype)[_A,_A,:]*E;B=B+A.mlp_scale.to(dtype=B.dtype)[_A,_A,:]*A.mlp(A.mlp_norm(B)*A.ln_scale_factor);return B
class _HessianGPT(nn.Module):
	def __init__(A,vocab_size,num_layers,model_dim,num_heads,num_kv_heads,mlp_mult,tie_embeddings,logit_softcap,rope_base,qk_gain_init,ngram_buckets=0,ngram_heads=2,ngram_orders=2,ngram_dim_per_head=32,xsa_last_n=0,rope_dims=0,ln_scale=_D,ve_enabled=_D,ve_dim=128,ve_layers=_V):
		K=xsa_last_n;J=ngram_buckets;I=rope_base;H=tie_embeddings;G=num_kv_heads;F=rope_dims;E=num_heads;D=vocab_size;C=num_layers;B=model_dim;super().__init__();A.tie_embeddings=H;A.logit_softcap=logit_softcap;A.num_layers=C;A.tok_emb=nn.Embedding(D,B);A.bigram=EngramLite(J,ngram_heads,ngram_orders,ngram_dim_per_head,B)if J>0 else _A;A.smear=SmearGate(B);A.num_encoder_layers=C//2;A.num_decoder_layers=C-A.num_encoder_layers;A.num_skip_weights=min(A.num_encoder_layers,A.num_decoder_layers);A.skip_weights=nn.Parameter(torch.ones(A.num_skip_weights,B,dtype=torch.float32));A.skip_gates=nn.Parameter(torch.zeros(A.num_skip_weights,B,dtype=torch.float32));A.blocks=nn.ModuleList([_HessianBlock(B,E,G,mlp_mult,I,qk_gain_init,layer_idx=A,ln_scale=ln_scale)for A in range(C)])
		if F>0:
			M=B//E
			for L in A.blocks:L.attn.rope_dims=F;L.attn.rotary=Rotary(M,base=I,train_seq_len=1024,rope_dims=F)
		if K>0:
			for N in range(max(0,C-K),C):A.blocks[N].attn.use_xsa=_B
		O=G*(B//E);A.ve_layer_indices=[int(A)for A in ve_layers.split(',')if A.strip()]if ve_enabled else[]
		if A.ve_layer_indices:A.ve_shared=ValueEmbedding(D,ve_dim,O);A.ve_layer_scales=nn.ParameterList([nn.Parameter(torch.ones(1,dtype=torch.float32))for A in A.ve_layer_indices])
		else:A.ve_shared=_A;A.ve_layer_scales=nn.ParameterList()
		A.final_norm=RMSNorm();A.lm_head=_A if H else CastedLinear(B,D,bias=_D)
	def _get_ve(A,layer_idx,input_ids,ve_cache):
		D=layer_idx;C='ve';B=ve_cache
		if A.ve_shared is _A or D not in A.ve_layer_indices:return
		if C not in B:B[C]=A.ve_shared(input_ids)
		E=A.ve_layer_indices.index(D);return B[C]*A.ve_layer_scales[E].to(dtype=B[C].dtype)
	def forward(B,input_ids,target_ids):
		D=input_ids;A=B.tok_emb(D)
		if B.bigram is not _A:A=A+B.bigram(D)
		A=F.rms_norm(A,(A.size(-1),));A=B.smear(A);H=A;E=[];I={}
		for C in range(B.num_encoder_layers):G=B._get_ve(C,D,I);A=B.blocks[C](A,H,v_embed=G);E.append(A)
		for C in range(B.num_decoder_layers):
			J=B.num_encoder_layers+C
			if E:L=E.pop();M=torch.sigmoid(B.skip_gates[C].to(dtype=A.dtype))[_A,_A,:];N=B.skip_weights[C].to(dtype=A.dtype)[_A,_A,:]*L;A=torch.lerp(N,A,M)
			G=B._get_ve(J,D,I);A=B.blocks[J](A,H,v_embed=G)
		A=B.final_norm(A);K=A.reshape(-1,A.size(-1));O=target_ids.reshape(-1);P=F.linear(K,B.tok_emb.weight)if B.tie_embeddings else B.lm_head(K);Q=B.logit_softcap*torch.tanh(P/B.logit_softcap);return F.cross_entropy(Q.float(),O,reduction='mean')
def collect_hessians(hessian_model,train_loader,args,device,grad_accum_steps,num_batches=256):
	G=num_batches;B=hessian_model;A={};H=[]
	for(C,D)in B.named_modules():
		if isinstance(D,CastedLinear):
			I=C+'.weight';J=D.weight.shape[1];A[I]=torch.zeros(J,J,dtype=torch.float32,device=device)
			def L(pname):
				def B(mod,inp,out):
					B=inp[0].detach()
					if B.ndim==3:B=B.reshape(-1,B.shape[-1])
					A[pname]+=(B.T@B).float()
				return B
			E=D.register_forward_hook(L(I));H.append(E)
	B.eval()
	with torch.inference_mode(),torch.autocast(device_type=_J,dtype=torch.bfloat16):
		for O in range(G):M,N=train_loader.next_batch(args.train_batch_tokens,args.train_seq_len,grad_accum_steps);B(M,N)
	for E in H:E.remove()
	K=dist.get_world_size()if dist.is_available()and dist.is_initialized()else 1
	for C in A:
		F=A[C]
		if K>1:dist.all_reduce(F,op=dist.ReduceOp.SUM)
		F/=G*K;A[C]=F.cpu()
	B.train();return A
def _bits_to_range(bits):return-(1<<bits-1),(1<<bits-1)-1
_MP_BYTES_PER_PARAM_INT5=.46
_MP_COST_PER_EXTRA_BIT=.24
_MP_NON_WEIGHT_COMPRESS=.55
_MP_PRUNE_HEADROOM_FRAC=.02
def _allocate_bits_mixed(hessian_map,state_dict,target_bytes=16000000,code_bytes=0):
	R=state_dict;Q=hessian_map;C=target_bytes;S={};H={};L={}
	for(B,T)in Q.items():
		b=float(torch.trace(T).item())/T.shape[0]
		if not B.startswith('blocks.'):continue
		c=B.index('.',7);d=int(B[7:c]);e=_b if _t in B else _a if _Z in B else'other';A=f"layer.{d}.{e}";S.setdefault(A,[]).append(b);L[B]=A;U=R.get(B)
		if U is not _A:H[A]=H.get(A,0)+U.numel()
	M={B:sum(A)/len(A)for(B,A)in S.items()};D=sorted(M.items(),key=lambda x:x[1],reverse=_B);f=sum(H.values());g=sum(A.numel()*A.element_size()for(B,A)in R.items()if B not in Q);E=code_bytes+int(g*_MP_NON_WEIGHT_COMPRESS)+int(f*_MP_BYTES_PER_PARAM_INT5);I=int(C*(_C-_MP_PRUNE_HEADROOM_FRAC))-E
	if I<=0:J={A:5 for A in L};N=[(A,5,M[A])for(A,B)in D];O={_c:E/1e6,_d:_E,_e:E/1e6,_f:C/1e6,_g:_E,_h:C-E,'warning':'budget_exhausted'};return J,N,O
	F={A:5 for(A,B)in D};G=0
	if D:
		P=D[0][0];K=H.get(P,0);V=int(K*_MP_COST_PER_EXTRA_BIT*2);W=int(K*_MP_COST_PER_EXTRA_BIT*1)
		if K>0 and V<=I:F[P]=7;G+=V
		elif K>0 and W<=I:F[P]=6;G+=W
	for(A,i)in D:
		if F[A]>5:continue
		X=H.get(A,0)
		if X==0:continue
		Y=int(X*_MP_COST_PER_EXTRA_BIT)
		if G+Y<=I:F[A]=6;G+=Y
	J={}
	for(h,A)in L.items():J[h]=F[A]
	N=[(A,F[A],M[A])for(A,B)in D];Z=E+G;a=int(C*_MP_PRUNE_HEADROOM_FRAC);O={_c:E/1e6,_d:G/1e6,_e:Z/1e6,_f:C/1e6,_g:a/1e3,_h:C-Z-a};return J,N,O
def mixed_quantize_int6(state_dict,int6_cats,hessians=_A,bit_allocation=_A,gptq_damp=.01,block_size=128,col_order=_M,single_pass=_D):
	L='type';H=bit_allocation;G=hessians;C={};D={}
	for(A,M)in state_dict.items():
		B=M.detach().cpu().contiguous();N=_classify_param(A)
		if not B.is_floating_point()or B.numel()<=65536 or A==_m:C[A]=B.to(torch.float16)if B.is_floating_point()else B;D[A]=_R;continue
		if any(B in A for B in CONTROL_TENSOR_NAME_PATTERNS):C[A]=B.float();D[A]=_y;continue
		if N in int6_cats and B.ndim>=1:
			I=H.get(A,6)if H else 6;P,O=_bits_to_range(I);J=O;K=G.get(A)if G else _A
			if K is not _A:E,F=quantize_int6_gptq(B,hessian=K,clip_range=J,block_size=block_size,damp_factor=gptq_damp,col_order=col_order,single_pass=single_pass)
			else:E,F=quantize_int6_per_row(B,clip_range=J)
			C[A+_T]=E;C[A+_U]=F;D[A]={L:f"int{I}"}
		else:E,F=quantize_float_tensor(B);C[A+_T]=E;C[A+_U]=F;D[A]={L:'int8'}
	return C,D
def dequantize_mixed_int6(result,meta,template_sd):
	F=result;B={}
	for(A,I)in template_sd.items():
		H=meta.get(A)
		if H is _A:continue
		C=I.dtype
		if H in(_R,_y,'passthrough_fp16'):
			D=F[A]
			if D.dtype==torch.float16 and C in(torch.float32,torch.bfloat16):D=D.to(C)
			B[A]=D;continue
		E,G=F[A+_T],F[A+_U]
		if G.ndim>0:B[A]=(E.float()*G.float().view(E.shape[0],*[1]*(E.ndim-1))).to(C)
		else:B[A]=(E.float()*float(G.item())).to(C)
	return B
def main():
	BD='final_model.int6.ptz';BC='sd_cpu';BB='hessians';BA='unbanked_sd';B9='final_model.pt';B8='WORLD_SIZE';U='base_lr';p=Path(__file__).read_text(encoding=_I);A=Hyperparameters();Z=int(os.environ.get('RANK','0'));I=int(os.environ.get(B8,_F));BE=int(os.environ.get('LOCAL_RANK','0'));P='RANK'in os.environ and B8 in os.environ and I>1
	if I<=0:raise ValueError(f"WORLD_SIZE must be positive, got {I}")
	if 8%I!=0:raise ValueError(f"WORLD_SIZE={I} must divide 8 so grad_accum_steps stays integral")
	Q=8//I;Ai=_C/Q
	if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
	F=torch.device(_J,BE);torch.cuda.set_device(F)
	if P:dist.init_process_group(backend='nccl',device_id=F);dist.barrier()
	q=Z==0;torch.backends.cuda.matmul.allow_tf32=_B;torch.backends.cudnn.allow_tf32=_B;from torch.backends.cuda import enable_cudnn_sdp as BF,enable_flash_sdp as BG,enable_math_sdp as BH,enable_mem_efficient_sdp as BI;BF(_B);BG(_B);BI(_B);BH(_B);x=_A
	if q:os.makedirs('logs',exist_ok=_B);x=f"logs/{A.run_id}.txt";print(x)
	def B(msg,console=_B):
		if not q:return
		if console:print(msg)
		if x is not _A:
			with open(x,'a',encoding=_I)as A:print(msg,file=A)
	B(p,console=_D);B('='*100,console=_D);B(f"Running Python {sys.version}",console=_D);B(f"Running PyTorch {torch.__version__}",console=_D);B(subprocess.run(['nvidia-smi'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=_B,check=_D).stdout,console=_D);B('='*100,console=_D);random.seed(A.seed);np.random.seed(A.seed);torch.manual_seed(A.seed);torch.cuda.manual_seed_all(A.seed)
	if not A.tokenizer_path.endswith('.model'):raise ValueError(f"Script only setup for SentencePiece .model file: {A.tokenizer_path}")
	AD=spm.SentencePieceProcessor(model_file=A.tokenizer_path)
	if int(AD.vocab_size())!=A.vocab_size:raise ValueError(f"VOCAB_SIZE={A.vocab_size} does not match tokenizer vocab_size={int(AD.vocab_size())}")
	Aj=Path(A.data_path).resolve();BJ=len(list(Aj.glob(_i)));AE=A.eval_seq_len if A.eval_seq_len>0 else A.train_seq_len;BK=max(A.train_seq_len,AE);y=load_validation_tokens(A.val_files,BK);AF,AG,AH=build_sentencepiece_luts(AD,A.vocab_size,F);B(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={A.tokenizer_path}");B(f"train_loader:dataset:{Aj.name} train_shards:{BJ}");B(f"val_loader:shards pattern={A.val_files} tokens:{y.numel()-1}")
	if not A.load_snapshot:
		CastedLinear._qat_enabled=_D;CastedLinear._qat_soft_round=_D;CastedLinear._qat_clip_pct=A.qat_clip_pct;C=GPT(vocab_size=A.vocab_size,num_layers=A.num_layers,model_dim=A.model_dim,num_heads=A.num_heads,num_kv_heads=A.num_kv_heads,mlp_mult=A.mlp_mult,tie_embeddings=A.tie_embeddings,tied_embed_init_std=A.tied_embed_init_std,logit_softcap=A.logit_softcap,rope_base=A.rope_base,qk_gain_init=A.qk_gain_init,ngram_buckets=A.ngram_buckets,ngram_heads=A.ngram_heads,ngram_orders=A.ngram_orders,ngram_dim_per_head=A.ngram_dim_per_head,xsa_last_n=A.xsa_last_n,rope_dims=A.rope_dims,ln_scale=A.ln_scale,ve_enabled=A.ve_enabled,ve_dim=A.ve_dim,ve_layers=A.ve_layers).to(F).bfloat16();C.qo_bank.data=C.qo_bank.data.float();C.kv_bank.data=C.kv_bank.data.float();C.mlp_up_bank.data=C.mlp_up_bank.data.float();C.mlp_down_bank.data=C.mlp_down_bank.data.float()
		for Ak in C.modules():
			if isinstance(Ak,CastedLinear):Ak.float()
		restore_low_dim_params_to_fp32(C);global zeropower_via_newtonschulz5,_post_ns_normalize;zeropower_via_newtonschulz5=torch.compile(zeropower_via_newtonschulz5);_post_ns_normalize=torch.compile(_post_ns_normalize);BL=torch.compile(C,dynamic=_D,fullgraph=_B);AI=BL;BM=[C.qo_bank,C.kv_bank,C.mlp_up_bank,C.mlp_down_bank];BN=list(C.blocks.named_parameters());V=[A for(B,A)in BN if A.ndim<2 or any(A in B for A in CONTROL_TENSOR_NAME_PATTERNS)]
		if C.skip_weights.numel()>0:V.append(C.skip_weights)
		if hasattr(C,'skip_gates')and C.skip_gates.numel()>0:V.append(C.skip_gates)
		V.append(C.smear.gate)
		if C.bigram is not _A:V.append(C.bigram.ngram_gate)
		a=A.tied_embed_lr if A.tie_embeddings else A.embed_lr;AJ=[{_G:[C.tok_emb.weight],_H:a,U:a}]
		if C.bigram is not _A:AJ.append({_G:[C.bigram.embed.weight],_H:a,U:a})
		z=_A
		if C.ve_shared is not _A:
			AJ.append({_G:[C.ve_shared.embed.weight],_H:a,U:a})
			if C.ve_shared.proj is not _A:z=C.ve_shared.proj.weight
			V.append(C.ve_shared.scale)
			for A0 in C.ve_layer_scales:V.append(A0)
		r=torch.optim.AdamW(AJ,betas=(A.embed_beta1,A.beta2),eps=A.adam_eps,weight_decay=A.adam_wd,fused=_B);W=Muon(BM,lr=A.matrix_lr,momentum=A.muon_momentum,backend_steps=A.muon_backend_steps,weight_decay=A.muon_wd,post_norm=A.muon_post_norm)
		for M in W.param_groups:M[U]=A.matrix_lr
		AK=torch.optim.AdamW([{_G:V,_H:A.scalar_lr,U:A.scalar_lr}],betas=(A.beta1,A.beta2),eps=A.adam_eps,weight_decay=A.adam_wd,fused=_B);X=list(r.param_groups[0][_G])
		for BO in r.param_groups[1:]:X.extend(BO[_G])
		X.extend(V);R=[r,W,AK]
		if C.bigram is not _A and C.bigram.proj is not _A:
			Al=Muon([C.bigram.proj.weight],lr=A.matrix_lr,momentum=A.muon_momentum,backend_steps=A.muon_backend_steps,weight_decay=A.muon_wd,post_norm=A.muon_post_norm)
			for M in Al.param_groups:M[U]=A.matrix_lr
			R.append(Al);X.append(C.bigram.proj.weight)
		if z is not _A:
			Am=Muon([z],lr=A.matrix_lr,momentum=A.muon_momentum,backend_steps=A.muon_backend_steps,weight_decay=A.muon_wd,post_norm=A.muon_post_norm)
			for M in Am.param_groups:M[U]=A.matrix_lr
			R.append(Am);X.append(z)
		A1=_A
		if C.lm_head is not _A:A1=torch.optim.Adam([{_G:[C.lm_head.weight],_H:A.head_lr,U:A.head_lr}],betas=(A.head_beta1,A.beta2),eps=A.adam_eps,fused=_B);X.append(C.lm_head.weight);R.append(A1)
		AL=[A.numel()for A in X];A2=torch.zeros(sum(AL),device=F,dtype=torch.float32)if P else _A;BP=sum(A.numel()for A in C.parameters());B(f"model_params:{BP}");BQ=[A for(A,B)in enumerate(C.blocks)if B.attn.use_xsa];B(f"XSA:last_{A.xsa_last_n} active_layers:{BQ}");B(f"world_size:{I} grad_accum_steps:{Q}");B('sdp_backends:cudnn=True flash=True mem_efficient=True math=True');B(f"attention_mode:gqa num_heads:{A.num_heads} num_kv_heads:{A.num_kv_heads}");B(f"tie_embeddings:{A.tie_embeddings} embed_lr:{a} head_lr:{A.head_lr if C.lm_head is not _A else _E} matrix_lr:{A.matrix_lr} scalar_lr:{A.scalar_lr}");B(f"train_batch_tokens:{A.train_batch_tokens} train_seq_len:{A.train_seq_len} iterations:{A.iterations} warmup_steps:{A.warmup_steps} max_wallclock_seconds:{A.max_wallclock_seconds:.3f}");B(f"muon:post_norm:{A.muon_post_norm} embed_beta1:{A.embed_beta1} head_beta1:{A.head_beta1}");B(f"seed:{A.seed}");AM=DistributedTokenLoader(A.train_files,Z,I,F)
		def s():
			for A in R:A.zero_grad(set_to_none=_B)
		J=1e3*A.max_wallclock_seconds if A.max_wallclock_seconds>0 else _A
		if J is not _A and A.gptq_calib_batches>0 and A.gptq_reserve_ms>0:J-=A.gptq_reserve_ms;B(f"gptq:reserving {A.gptq_reserve_ms:.0f}ms from training budget (effective cap: {J:.0f}ms)")
		def BR(step,elapsed_ms):
			C=elapsed_ms;B=step
			if A.warmdown_iters<=0:return _C
			if J is _A:F=max(A.iterations-A.warmdown_iters,0);return max((A.iterations-B)/max(A.warmdown_iters,1),A.lr_floor)if F<=B<A.iterations else _C
			G=C/max(B,1);D=min(A.warmdown_iters*G,J);E=max(J-C,_E);return max(E/max(D,1e-09),A.lr_floor)if E<=D else _C
		if A.warmup_steps>0:
			BS={A:B.detach().cpu().clone()for(A,B)in C.state_dict().items()};BT=[copy.deepcopy(A.state_dict())for A in R];AI.train()
			for AN in range(A.warmup_steps):
				s()
				for BU in range(Q):
					AO,AP=AM.next_batch(A.train_batch_tokens,A.train_seq_len,Q)
					with torch.autocast(device_type=_J,dtype=torch.bfloat16,enabled=_B):BV=AI(AO,AP)
					(BV*Ai).backward()
				if P:
					for D in C.parameters():
						if D.grad is not _A:dist.all_reduce(D.grad,op=dist.ReduceOp.AVG)
				for K in R:K.step()
				s()
				if A.warmup_steps<=20 or(AN+1)%10==0 or AN+1==A.warmup_steps:B(f"warmup_step:{AN+1}/{A.warmup_steps}")
			C.load_state_dict(BS,strict=_B)
			for(K,BW)in zip(R,BT,strict=_B):K.load_state_dict(BW)
			if W._built:
				for b in W._bank_meta:b[_Y].zero_()
			s();AM=DistributedTokenLoader(A.train_files,Z,I,F)
		c=_A;t=0;i=_A;AQ=_A
		if A.ema_enabled:i={A:B.data.detach().float().clone()for(A,B)in C.named_parameters()};AQ=[(i[A],B)for(A,B)in C.named_parameters()]
		j=_E;AR=0;k=0;A3=torch.zeros((),device=F);AS=torch.zeros(1,device=F,dtype=torch.int32)if P else _A;N=_A;torch.cuda.synchronize();AT=time.perf_counter();E=0
		while _B:
			BX=E==A.iterations or N is not _A and E>=N
			if BX:
				torch.cuda.synchronize();j+=1e3*(time.perf_counter()-AT);B(f"step:{E}/{A.iterations} train_time:{j:.0f}ms step_avg:{j/max(E,1):.2f}ms")
				if N is not _A and E<A.iterations:B(f"stopping_early: wallclock_cap train_time:{j:.0f}ms step:{E}/{A.iterations}")
				break
			AU=j+1e3*(time.perf_counter()-AT);A4=BR(E,AU)
			if A.late_qat and A.qat_threshold>0 and not CastedLinear._qat_enabled and A4<A.qat_threshold:
				CastedLinear._qat_enabled=_B;AR=E
				if N is not _A:k=max(N-E,1)
				elif J is not _A:BY=AU/max(E,1);k=max(int((J-AU)/max(BY,1e-09)),1)
				else:k=max(A.iterations-E,1)
				if A.soft_round_qat:CastedLinear._qat_soft_round=_B;CastedLinear._qat_soft_alpha=torch.tensor(_C,device=F)
				B(f"late_qat:enabled step:{E} scale:{A4:.4f} soft_round={A.soft_round_qat} est_steps:{k}")
			elif CastedLinear._qat_soft_round:
				if N is not _A:k=max(N-AR,1)
				BZ=min((E-AR)/k,_C)
				with torch.no_grad():CastedLinear._qat_soft_alpha.fill_(_C+15.*BZ)
			s();A3.zero_()
			for BU in range(Q):
				AO,AP=AM.next_batch(A.train_batch_tokens,A.train_seq_len,Q)
				with torch.autocast(device_type=_J,dtype=torch.bfloat16,enabled=_B):An=AI(AO,AP)
				A3+=An.detach();(An*Ai).backward()
			A3/=Q;Ao=min(E/A.muon_momentum_warmup_steps,_C)if A.muon_momentum_warmup_steps>0 else _C;Ba=(1-Ao)*A.muon_momentum_warmup_start+Ao*A.muon_momentum
			for M in W.param_groups:M[_k]=Ba
			for K in R:
				for M in K.param_groups:M[_H]=M[U]*A4
			if A.grad_clip_norm>0:torch.nn.utils.clip_grad_norm_(C.parameters(),A.grad_clip_norm)
			W.launch_reduce_scatters()
			if P:
				S=0
				for(D,l)in zip(X,AL):
					if D.grad is not _A:A2[S:S+l].copy_(D.grad.reshape(-1))
					else:A2[S:S+l].zero_()
					S+=l
				dist.all_reduce(A2,op=dist.ReduceOp.AVG);S=0
				for(D,l)in zip(X,AL):
					if D.grad is not _A:D.grad.copy_(A2[S:S+l].reshape_as(D.grad))
					S+=l
			r.step();AK.step()
			for K in R:
				if K is not W and K is not r and K is not AK:
					if A1 is not _A and K is A1:K.step()
					elif isinstance(K,Muon):K.step()
			W.step();s()
			if AQ is not _A:
				Bb=_C-A.ema_decay
				with torch.no_grad():
					for(Bc,D)in AQ:Bc.lerp_(D.data if D.data.dtype==torch.float32 else D.data.float(),Bb)
			E+=1;u=j+1e3*(time.perf_counter()-AT)
			if A.swa_enabled and A4<A.swa_threshold and E%A.swa_every==0:
				if c is _A:c={A:B.data.detach().float().clone()for(A,B)in C.named_parameters()};t=1;B(f"swa:start step:{E}")
				else:
					for(O,D)in C.named_parameters():c[O].add_(D.data if D.data.dtype==torch.float32 else D.data.float())
					t+=1
			Bd=A.train_log_every>0 and(E<=10 or E%A.train_log_every==0 or N is not _A)
			if Bd:B(f"step:{E}/{A.iterations} train_loss:{A3.item():.4f} train_time:{u:.0f}ms step_avg:{u/E:.2f}ms")
			AV=J is not _A and u>=J
			if P and J is not _A:AS.fill_(int(AV));dist.all_reduce(AS,op=dist.ReduceOp.MAX);AV=bool(AS.item())
			if N is _A and AV:N=E
		B(f"peak memory allocated: {torch.cuda.max_memory_allocated()//1024//1024} MiB reserved: {torch.cuda.max_memory_reserved()//1024//1024} MiB")
		if A.swa_enabled and c is not _A and t>1:
			B(f"swa:applying averaged {t} checkpoints (source=raw)")
			with torch.no_grad():
				for(O,D)in C.named_parameters():
					if O in c:D.data.copy_((c[O]/t).to(dtype=D.dtype))
			del c
		elif i is not _A:
			B('ema:applying EMA weights')
			with torch.no_grad():
				for(O,D)in C.named_parameters():
					if O in i:D.data.copy_(i[O].to(dtype=D.dtype))
			del i
		Be=C.state_dict();Ap=Be
		if q:torch.save(Ap,B9);Bf=os.path.getsize(B9);AW=len(p.encode(_I));B(f"Serialized model: {Bf} bytes");B(f"Code size: {AW} bytes")
		A5={A:B.detach().cpu()for(A,B)in Ap.items()};d=_unbank_state_dict(A5,A.num_layers);Bg=time.perf_counter();B(f"gptq:building non-banked model for Hessian collection...");m=_HessianGPT(vocab_size=A.vocab_size,num_layers=A.num_layers,model_dim=A.model_dim,num_heads=A.num_heads,num_kv_heads=A.num_kv_heads,mlp_mult=A.mlp_mult,tie_embeddings=A.tie_embeddings,logit_softcap=A.logit_softcap,rope_base=A.rope_base,qk_gain_init=A.qk_gain_init,ngram_buckets=A.ngram_buckets,ngram_heads=A.ngram_heads,ngram_orders=A.ngram_orders,ngram_dim_per_head=A.ngram_dim_per_head,xsa_last_n=A.xsa_last_n,rope_dims=A.rope_dims,ln_scale=A.ln_scale,ve_enabled=A.ve_enabled,ve_dim=A.ve_dim,ve_layers=A.ve_layers).to(F).bfloat16()
		for b in m.modules():
			if isinstance(b,CastedLinear):b.float()
		restore_low_dim_params_to_fp32(m);m.load_state_dict({A:B.to(F)for(A,B)in d.items()if A in m.state_dict()},strict=_D);B(f"gptq:calibrating with {A.gptq_calib_batches} batches...");Bh=DistributedTokenLoader(A.train_files,Z,I,F);e=collect_hessians(m,Bh,A,F,Q,num_batches=A.gptq_calib_batches);B(f"gptq:collected hessians for {len(e)} layers");Aq=1e3*(time.perf_counter()-Bg);Bi=J+A.gptq_reserve_ms if J is not _A else 0;B(f"gptq:budget_check train:{u:.0f}ms + gptq:{Aq:.0f}ms = {u+Aq:.0f}ms (budget:{Bi:.0f}ms)");del m;torch.cuda.empty_cache()
		if A.snapshot_post_hessian:
			if q:A6=os.environ.get('SNAPSHOT_PATH','snapshot_post_hessian.pt');B(f"snapshot:saving to {A6}...");torch.save({BA:d,BB:{A:B.cpu()for(A,B)in e.items()},BC:A5},A6);Bj=os.path.getsize(A6)/1e6;B(f"snapshot:saved {Bj:.1f}MB — exiting (use LOAD_SNAPSHOT={A6} to resume compression)")
			if P:dist.barrier();dist.destroy_process_group()
			return
	if A.load_snapshot:B(f"snapshot:loading from {A.load_snapshot}...");A7=torch.load(A.load_snapshot,map_location=_Q,weights_only=_B);d=A7[BA];e={A:B.to(F)for(A,B)in A7[BB].items()};A5=A7[BC];del A7;B(f"snapshot:restored {len(d)} unbanked params, {len(e)} hessians")
	AX=_A
	if A.mixed_precision and e:
		Bk=len(p.encode(_I));AX,Ar,n=_allocate_bits_mixed(e,d,target_bytes=A.target_bytes_limit,code_bytes=Bk);B(f"mixed_precision:estimate base={n[_c]:.2f}MB + promoted={n[_d]:.2f}MB = {n[_e]:.2f}MB (budget={n[_f]:.1f}MB, headroom={n[_g]:.0f}KB, prune_room={n[_h]:+.0f}B)");Bl=sum(1 for(A,B,A)in Ar if B>5)
		for(Bm,Bn,Bo)in Ar:B(f"mixed_precision: {Bm} -> int{Bn} (sensitivity={Bo:.4e})")
		AY={}
		for As in AX.values():AY[As]=AY.get(As,0)+1
		B(f"mixed_precision: {' '.join(f'int{A}:{B}'for(A,B)in sorted(AY.items()))} ({Bl} groups promoted)")
	Y,AZ=mixed_quantize_int6(d,{_a,_b},hessians=e,bit_allocation=AX,gptq_damp=A.gptq_damp,block_size=A.gptq_block_size,col_order=A.gptq_col_order,single_pass=A.gptq_single_pass);f=A.target_bytes_limit;Bp=len(p.encode(_I));G=[]
	for(O,Bq)in AZ.items():
		if not isinstance(Bq,dict):continue
		Aa,At=O+_T,O+_U
		if Aa not in Y or At not in Y:continue
		g,A0=Y[Aa],Y[At]
		if A0.ndim>0:
			A8=(g.abs()<=2)&(g.abs()>0)
			if A8.any():
				Br=torch.arange(g.shape[0]).unsqueeze(1).expand_as(g)[A8];Bs=torch.arange(g.numel()).reshape(g.shape)[A8];Bt=g.abs()[A8].float();Bu=A0.float()[Br].pow(2)*Bt.pow(2)
				for(Bv,Bw)in zip(Bs.tolist(),Bu.tolist()):G.append((Aa,Bv,Bw))
	if G:
		G.sort(key=lambda x:x[2]);o={};A9={}
		for(Au,(T,Av,C7))in enumerate(G):
			if T in o:o[T].append(Au);A9[T].append(Av)
			else:o[T]=[Au];A9[T]=[Av]
		Ab={}
		for T in o:Ab[T]=torch.tensor(o[T],dtype=torch.long),torch.tensor(A9[T],dtype=torch.long)
		del o,A9
		def Bx(qr,qm,fast=_D):
			C=io.BytesIO();torch.save({'w':qr,'m':qm},C);A=C.getvalue()
			if _BYTE_SHUFFLE:A=_byte_shuffle(A,_BYTE_SHUFFLE_STRIDE)
			if fast:B=zlib.compress(A,1)
			elif _COMPRESSOR==_K:import brotli as D;B=D.compress(A,quality=11)
			elif _COMPRESSOR==_L:B=lzma.compress(A,preset=9)
			else:B=zlib.compress(A,9)
			return len(B)+Bp
		def h(n,fast=_D):
			n=min(n,len(G));A=dict(Y)
			for(B,(D,E))in Ab.items():
				C=int(torch.searchsorted(D,n).item())
				if C>0:A[B]=Y[B].clone();A[B].view(-1)[E[:C]]=0
			return Bx(A,AZ,fast=fast)
		def Aw(n):
			n=min(n,len(G))
			for(B,(C,D))in Ab.items():
				A=int(torch.searchsorted(C,n).item())
				if A>0:Y[B].view(-1)[D[:A]]=0
		AA=h(0);B(f"selective_prune: {len(G)} +/-1,+/-2 candidates, unpruned={AA/1048576:.2f}MB target={f/1e6:.2f}MB")
		if AA<=f:B('selective_prune: already fits, no pruning needed')
		else:
			Ax=h(0,fast=_B);By=h(len(G),fast=_B);Ac=h(len(G));B(f"selective_prune: full prune={Ac/1048576:.2f}MB")
			if Ac>f:B('selective_prune: even full prune not enough, applying all');Aw(len(G))
			else:
				Bz=Ax-By;B_=AA-Ac;Ay=B_/max(Bz,1);Az=Ax-int((AA-f)/max(Ay,.01));B(f"selective_prune: fast/real ratio={Ay:.3f} fast_target={Az/1048576:.2f}MB");L,Ad=0,len(G)
				while L<Ad:
					Ae=(L+Ad)//2;C0=h(Ae,fast=_B)
					if C0<=Az:Ad=Ae
					else:L=Ae+1
				Af=h(L)
				if Af>f:
					while L<len(G)and Af>f:L+=max(1,len(G)//200);L=min(L,len(G));Af=h(L)
				B(f"selective_prune: pruning {L}/{len(G)} values ({100*L/max(len(G),1):.1f}%) to fit {f/1e6:.2f}MB");Aw(L)
	A_=io.BytesIO();torch.save({'w':Y,'m':AZ},A_);v=A_.getvalue()
	if _BYTE_SHUFFLE:v=_byte_shuffle(v,_BYTE_SHUFFLE_STRIDE)
	if _COMPRESSOR==_K:import brotli;AB=brotli.compress(v,quality=11)
	elif _COMPRESSOR==_L:AB=lzma.compress(v,preset=9)
	else:AB=zlib.compress(v,9)
	if q:
		with open(BD,'wb')as Ag:Ag.write(AB)
		B0=len(AB);AW=len(p.encode(_I));B(f"Serialized model int6+{_COMPRESSOR}: {B0} bytes");B(f"Total submission size int6+{_COMPRESSOR}: {B0+AW} bytes")
	if P:dist.barrier()
	with open(BD,'rb')as Ag:Ah=Ag.read()
	if _COMPRESSOR==_K:import brotli;w=brotli.decompress(Ah)
	elif _COMPRESSOR==_L:w=lzma.decompress(Ah)
	else:w=zlib.decompress(Ah)
	w=_byte_unshuffle(w);B1=torch.load(io.BytesIO(w),map_location=_Q,weights_only=_D);C1=dequantize_mixed_int6(B1['w'],B1['m'],d);C2=_rebank_state_dict(C1,A.num_layers,A5);H=GPT(vocab_size=A.vocab_size,num_layers=A.num_layers,model_dim=A.model_dim,num_heads=A.num_heads,num_kv_heads=A.num_kv_heads,mlp_mult=A.mlp_mult,tie_embeddings=A.tie_embeddings,tied_embed_init_std=A.tied_embed_init_std,logit_softcap=A.logit_softcap,rope_base=A.rope_base,qk_gain_init=A.qk_gain_init,ngram_buckets=A.ngram_buckets,ngram_heads=A.ngram_heads,ngram_orders=A.ngram_orders,ngram_dim_per_head=A.ngram_dim_per_head,xsa_last_n=A.xsa_last_n,rope_dims=A.rope_dims,ln_scale=A.ln_scale,ve_enabled=A.ve_enabled,ve_dim=A.ve_dim,ve_layers=A.ve_layers).to(F).bfloat16();H.qo_bank.data=H.qo_bank.data.float();H.kv_bank.data=H.kv_bank.data.float();H.mlp_up_bank.data=H.mlp_up_bank.data.float();H.mlp_down_bank.data=H.mlp_down_bank.data.float()
	for b in H.modules():
		if isinstance(b,CastedLinear):b.float()
	restore_low_dim_params_to_fp32(H);H.load_state_dict(C2,strict=_B);C3=torch.compile(H,dynamic=_D,fullgraph=_B);torch.cuda.synchronize();C4=time.perf_counter();B2,B3=eval_val(A,C3,Z,I,F,Q,y,AF,AG,AH,eval_seq_len=AE);torch.cuda.synchronize();B(f"final_int6_roundtrip val_loss:{B2:.4f} val_bpb:{B3:.4f} eval_time:{1e3*(time.perf_counter()-C4):.0f}ms");B(f"final_int6_roundtrip_exact val_loss:{B2:.8f} val_bpb:{B3:.8f}");AC=AE
	if A.eval_stride>0 and A.eval_stride<AC:torch.cuda.synchronize();C5=time.perf_counter();B4,B5=eval_val_sliding(A,H,Z,I,F,y,AF,AG,AH,stride=A.eval_stride,eval_seq_len=AC);torch.cuda.synchronize();B(f"final_int6_sliding_window val_loss:{B4:.4f} val_bpb:{B5:.4f} stride:{A.eval_stride} eval_time:{1e3*(time.perf_counter()-C5):.0f}ms");B(f"final_int6_sliding_window_exact val_loss:{B4:.8f} val_bpb:{B5:.8f}")
	if A.eval_stride!=64 and 64<AC:torch.cuda.synchronize();C6=time.perf_counter();B6,B7=eval_val_sliding(A,H,Z,I,F,y,AF,AG,AH,stride=64,eval_seq_len=AC);torch.cuda.synchronize();B(f"final_int6_sliding_window_s64 val_loss:{B6:.4f} val_bpb:{B7:.4f} stride:64 eval_time:{1e3*(time.perf_counter()-C6):.0f}ms");B(f"final_int6_sliding_window_s64_exact val_loss:{B6:.8f} val_bpb:{B7:.8f}")
	if P:dist.destroy_process_group()
if __name__=='__main__':main()