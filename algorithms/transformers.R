library(torch)
library(patchwork)
source("algorithms/functions.R") # helper functions within AI optimization  

# Setup GPU ---------------------------------------------------------------
dvc <- torch_device(if(cuda_is_available())"cuda" else "cpu")
dvc

# Hyper-parameters -----------------------------------------------------------------------------------------------------------------------------------------
args <- NULL
args$nb_nodes = 16L # TSP16
args$bsz = 512L # TSP20 TSP50
args$dim_emb = 128L
args$dim_ff = 512L
args$dim_input_nodes = 2L
args$nb_layers_encoder = 6L
args$nb_layers_decoder = 2L
args$nb_heads = 8L
args$nb_epochs = 50L
args$nb_batch_per_epoch = 100L
args$nb_batch_eval = 20L
# args$gpu_id = gpu_id
args$lr = 1e-4
args$tol = 1e-3
args$batchnorm = TRUE  # if batchnorm=True  than batch norm is used
#args$batchnorm = FALSE # if batchnorm=False than layer norm is used
args$max_len_PE = 1000L

# Functions -----------------------------------------------------------------------------------------------------------------------------------------------
calc_dist_batch <- function(base, sol){
  # base$gather(sol$unsqueeze(3)$expand(c(-1,-1,2)), dim=2)
  base$gather(sol$unsqueeze(3)$`repeat`(c(1,1,2)), dim=2) |>
    torch_split(1) |>
    lapply(\(x)-calc_dist4tnsr(x[-1])) |> 
    torch_stack(1)
}


# NN Encoder ----------------------------------------------------------------------------------------------------------------------------------------------
encoder <- nn_module(
  # Encoder network based on self-attention transformer
  # Inputs :  
  #   h of size      (bsz, nb_nodes+1, dim_emb)    batch of input cities
  # Outputs :  
  #   h of size      (bsz, nb_nodes+1, dim_emb)    batch of encoded cities
  #   score of size  (bsz, nb_nodes+1, nb_nodes+1) batch of attention scores
  
  classname = "Transformer encoder net",
  initialize = function(nb_layers, dim_emb, nb_heads, dim_ff, batchnorm){
    stopifnot(dim_emb %% nb_heads == 0) # check if dim_emb is divisible by nb_heads
    
    self$MHA_layers = seq_len(nb_layers) |> 
      lapply(\(x)nn_multihead_attention(dim_emb, nb_heads)) |> 
      nn_module_list() |> 
      with_device(device=dvc)
    
    self$linear1_layers = seq_len(nb_layers) |> 
      lapply(\(x)nn_linear(dim_emb, dim_ff)) |> 
      nn_module_list() |> 
      with_device(device=dvc)
    
    self$linear2_layers = seq_len(nb_layers) |> 
      lapply(\(x)nn_linear(dim_ff, dim_emb)) |> 
      nn_module_list() |> 
      with_device(device=dvc)
    
    if(batchnorm){
      self$norm1_layers = seq_len(nb_layers) |> 
        lapply(\(x)nn_batch_norm1d(dim_emb)) |> 
        nn_module_list() |> 
        with_device(device=dvc)
      
      self$norm2_layers = seq_len(nb_layers) |> 
        lapply(\(x)nn_batch_norm1d(dim_emb)) |> 
        nn_module_list() |> 
        with_device(device=dvc)
    }
    else{
      self$norm1_layers = seq_len(nb_layers) |> 
        lapply(\(x)nn_layer_norm(dim_emb)) |> 
        nn_module_list() |> 
        with_device(device=dvc)
      
      self$norm2_layers = seq_len(nb_layers) |> 
        lapply(\(x)nn_layer_norm(dim_emb)) |> 
        nn_module_list() |> 
        with_device(device=dvc)
    }
    
    self$nb_layers = nb_layers
    self$nb_heads = nb_heads
    self$batchnorm = batchnorm
  },
  forward = function(h){
    # browser()
    # Torch nn_multihead_attention() requires input tensor shape (seq_len, bsz, dim_emb)
    # seq_len - target sequence length
    # bsz - the batch size
    # dim_emb - the embedding
    h = h$transpose(1,2) # size(h)=(nb_nodes, bsz, dim_emb)  
    # L layers
    for (i in seq_len(self$nb_layers)){
      h_rc = h # residual connection, size(h_rc)=(nb_nodes, bsz, dim_emb)
      # c(h, score) %<-% self$MHA_layers[[i]](h, h, h) 
      MHA = self$MHA_layers[[i]](h, h, h) 
      h = MHA[[1]] # size(h)=(nb_nodes, bsz, dim_emb)
      score = MHA[[2]] # size(score)=(bsz, nb_nodes, nb_nodes)
      
      # add residual connection
      h = h_rc + h # size(h)=(nb_nodes, bsz, dim_emb)
      if(self$batchnorm){
        # Torch nn_batch_norm1d() requires input size (bsz, dim, seq_len) 
        h = h$permute(c(2,3,1)) |> #$contiguous()  # size(h)=(bsz, dim_emb, nb_nodes)
             self$norm1_layers[[i]]()              # size(h)=(bsz, dim_emb, nb_nodes)
        h = h$permute(c(3,1,2))#$contiguous()      # size(h)=(nb_nodes, bsz, dim_emb)
        # contiguous() makes sure that tensor is align witn memory structure for better performance
      }
      else{
        h = self$norm1_layers[[i]](h)        # size(h)=(nb_nodes, bsz, dim_emb) 
      }
      # feedforward
      h_rc = h # residual connection
      h = self$linear1_layers[[i]](h) |> 
        torch_relu() |> 
        self$linear2_layers[[i]]()
      h = h_rc + h # size(h)=(nb_nodes, bsz, dim_emb)
      
      if (self$batchnorm){
        h = h$permute(c(2,3,1)) |> #$contiguous() # size(h)=(bsz, dim_emb, nb_nodes)
             self$norm2_layers[[i]]()        # size(h)=(bsz, dim_emb, nb_nodes)
        h = h$permute(c(3,1,2))#$contiguous() # size(h)=(nb_nodes, bsz, dim_emb)
      }
      else{
        h = self$norm2_layers[[i]](h) # size(h)=(nb_nodes, bsz, dim_emb)
      }
    }
    # Transpose h
    h = h$transpose(1,2) # size(h)=(bsz, nb_nodes, dim_emb)
    list(h, score)
  }
)
# test encoder()
# encoder(3, 20, 2L, 4, TRUE)(torch_rand(3, 16, 20, device = dvc))
# enc_tst <- encoder(3, 20, 2L, 4, TRUE)
# profvis::profvis(enc_tst(torch_rand(3, 16, 20)))

# NN Attention ---------------------------------------------------------------------------------------------------------------------------------------------
my_MHA <- function(Q, K, V, nb_heads, mask=NULL, clip_value=NULL){
  # Compute multi-head attention (MHA) given a query Q, key K, value V and attention mask :
  #   h = Concat_{k=1}^nb_heads softmax(Q_k^T.K_k).V_k 
  # Note : We did not use nn.MultiheadAttention to avoid re-computing all linear transformations at each call.
  # Inputs : Q of size (bsz, 1, dim_emb)                batch of queries
  #          K of size (bsz, nb_nodes+1, dim_emb)       batch of keys
  #          V of size (bsz, nb_nodes+1, dim_emb)       batch of values
  #          mask of size (bsz, nb_nodes+1)             batch of masks of visited cities
  #          clip_value is a scalar 
  # Outputs : attn_output of size (bsz, 1, dim_emb)     batch of attention vectors
  #           attn_weights of size (bsz, 1, nb_nodes+1) batch of attention weights
  # 
  # c(bsz, nb_nodes, emd_dim) %<-% K$size()
  bsz = K$size(1)
  nb_nodes = K$size(2)
  emd_dim = K$size(3)  #  dim_emb must be divisable by nb_heads
  #browser()
  if (nb_heads>1){
    # PyTorch view requires contiguous dimensions for correct reshaping
    Q = Q$transpose(2,3)#$contiguous() # size(Q)=(bsz, dim_emb, 1)
    Q = Q$view(c(bsz*nb_heads, emd_dim %/% nb_heads, 1)) # size(Q)=(bsz*nb_heads, dim_emb//nb_heads, 1)
    Q = Q$transpose(2,3)#$contiguous() # size(Q)=(bsz*nb_heads, 1, dim_emb//nb_heads)
    K = K$transpose(2,3)$contiguous() # size(K)=(bsz, dim_emb, nb_nodes+1)
    K = K$view(c(bsz*nb_heads, emd_dim %/% nb_heads, nb_nodes)) # size(K)=(bsz*nb_heads, dim_emb//nb_heads, nb_nodes+1)
    K = K$transpose(2,3)#$contiguous() # size(K)=(bsz*nb_heads, nb_nodes+1, dim_emb//nb_heads)
    V = V$transpose(2,3)$contiguous() # size(V)=(bsz, dim_emb, nb_nodes+1)
    V = V$view(c(bsz*nb_heads, emd_dim %/% nb_heads, nb_nodes)) # size(V)=(bsz*nb_heads, dim_emb//nb_heads, nb_nodes+1)
    V = V$transpose(2,3)#$contiguous() # size(V)=(bsz*nb_heads, nb_nodes+1, dim_emb//nb_heads)
  }
  attn_weights = torch_bmm(Q, K$transpose(2,3))/ Q$size(-1)^0.5 # size(attn_weights)=(bsz*nb_heads, 1, nb_nodes+1)
  if(!is.null(clip_value)){attn_weights = clip_value * torch_tanh(attn_weights)}
  if(!is.null(mask)){
    if(nb_heads>1){
      mask = torch_repeat_interleave(mask, repeats=torch_tensor(nb_heads, device=dvc), dim=1) # size(mask)=(bsz*nb_heads, nb_nodes+1)
    } 
    attn_weights = attn_weights$masked_fill(mask$unsqueeze(2), -1e9) # size(attn_weights)=(bsz*nb_heads, 1, nb_nodes+1)
  }
  attn_weights = nnf_softmax(attn_weights, dim=-1) # size(attn_weights)=(bsz*nb_heads, 1, nb_nodes+1)
  attn_output = torch_bmm(attn_weights, V) # size(attn_output)=(bsz*nb_heads, 1, dim_emb//nb_heads)
  if(nb_heads>1){
    attn_output = attn_output$transpose(2,3)#$contiguous() # size(attn_output)=(bsz*nb_heads, dim_emb//nb_heads, 1)
    attn_output = attn_output$view(c(bsz, emd_dim, 1)) # size(attn_output)=(bsz, dim_emb, 1)
    attn_output = attn_output$transpose(2,3)#$contiguous() # size(attn_output)=(bsz, 1, dim_emb)
    attn_weights = attn_weights$view(c(bsz, nb_heads, 1, nb_nodes)) |># size(attn_weights)=(bsz, nb_heads, 1, nb_nodes+1)
      torch_mean(dim=2) # mean over the heads, size(attn_weights)=(bsz, 1, nb_nodes+1)
  }
  list(attn_output, attn_weights)
}
# test my_MHA()
# my_MHA(torch_rand(3, 1, 20,device=dvc), torch_rand(3, 16, 20,device=dvc), torch_rand(3, 16, 20,device=dvc), 2L, torch_randint(0,2, c(3,16), dtype=torch_bool(),device=dvc))
# mha_tst <- torch_rand(3, 1, 20)
# profvis::profvis(my_MHA(mha_tst, mha_tst, mha_tst, 2L, torch_randint(0,2, c(3,16), dtype=torch_bool())))


# NN Decoder ----------------------------------------------------------------------------------------------------------------------------------------------
decoder_layer <- nn_module(
  # Single decoder layer based on self-attention and query-attention
  # Inputs :  
  #   h_t of size      (bsz, 1, dim_emb)          batch of input queries
  #   K_att of size    (bsz, nb_nodes+1, dim_emb) batch of query-attention keys
  #   V_att of size    (bsz, nb_nodes+1, dim_emb) batch of query-attention values
  #   mask of size     (bsz, nb_nodes+1)          batch of masks of visited cities
  # Output :  
  #   h_t of size (bsz, nb_nodes+1)               batch of transformed queries
  
  classname = "AutoRegressive decoder layer",
  initialize = function(dim_emb, nb_heads){
    self$dim_emb = dim_emb
    self$nb_heads = nb_heads
    self$Wq_selfatt = nn_linear(dim_emb, dim_emb) |> with_device(device=dvc)
    self$Wk_selfatt = nn_linear(dim_emb, dim_emb) |> with_device(device=dvc)
    self$Wv_selfatt = nn_linear(dim_emb, dim_emb) |> with_device(device=dvc)
    self$W0_selfatt = nn_linear(dim_emb, dim_emb) |> with_device(device=dvc)
    self$W0_att = nn_linear(dim_emb, dim_emb) |> with_device(device=dvc)
    self$Wq_att = nn_linear(dim_emb, dim_emb) |> with_device(device=dvc)
    self$W1_MLP = nn_linear(dim_emb, dim_emb) |> with_device(device=dvc)
    self$W2_MLP = nn_linear(dim_emb, dim_emb) |> with_device(device=dvc)
    self$BN_selfatt = nn_layer_norm(dim_emb) |> with_device(device=dvc)
    self$BN_att = nn_layer_norm(dim_emb) |> with_device(device=dvc)
    self$BN_MLP = nn_layer_norm(dim_emb) |> with_device(device=dvc)
    self$K_sa = NULL
    self$V_sa = NULL
  },
  reset_selfatt_keys_values = function(){
    self$K_sa = NULL
    self$V_sa = NULL
  },
  forward = function(h_t, K_att, V_att, mask){
    # browser()
    bsz = h_t$size(1)
    h_t = h_t$view(c(bsz,1, self$dim_emb)) # size(h_t)=(bsz, 1, dim_emb)
    # embed the query for self-attention
    q_sa = self$Wq_selfatt(h_t) # size(q_sa)=(bsz, 1, dim_emb)
    k_sa = self$Wk_selfatt(h_t) # size(k_sa)=(bsz, 1, dim_emb)
    v_sa = self$Wv_selfatt(h_t) # size(v_sa)=(bsz, 1, dim_emb)
    # concatenate the new self-attention key and value to the previous keys and values
    if(is.null(self$K_sa)){
      self$K_sa = k_sa # size(self$K_sa)=(bsz, 1, dim_emb)
      self$V_sa = v_sa # size(self$V_sa)=(bsz, 1, dim_emb)
    }
    else{
      self$K_sa = torch_cat(list(self$K_sa, k_sa), dim=2)
      self$V_sa = torch_cat(list(self$V_sa, v_sa), dim=2)
    }
    # compute self-attention between nodes in the partial tour
    h_t = h_t + self$W0_selfatt(my_MHA(q_sa, self$K_sa, self$V_sa, self$nb_heads)[[1]]) # size(h_t)=(bsz, 1, dim_emb)
    h_t = self$BN_selfatt(h_t$squeeze()) |> # size(h_t)=(bsz, dim_emb)
      torch_unsqueeze(2) # size(h_t)=(bsz, 1, dim_emb)
 
    # compute attention between self-attention nodes and encoding nodes in the partial tour (translation process)
    q_a = self$Wq_att(h_t) # size(q_a)=(bsz, 1, dim_emb)
    h_t = h_t + self$W0_att(my_MHA(q_a, K_att, V_att, self$nb_heads, mask)[[1]]) # size(h_t)=(bsz, 1, dim_emb)
    h_t = self$BN_att(h_t) # size(h_t)=(bsz, 1, dim_emb)
    
    # MLP
    h_t = h_t + self$W2_MLP(torch_relu(self$W1_MLP(h_t)))
    self$BN_MLP(h_t$squeeze(2)) # size(h_t)=(bsz, dim_emb)
  }
)
# decoder_layer(20,2L)(torch_rand(3,1,20,device=dvc), torch_rand(3,16,20,device=dvc), torch_rand(3,16,20,device=dvc),torch_randint(0,2, c(3,16), dtype=torch_bool(), ,device=dvc))

decoder <- nn_module(
  # Decoder network based on self-attention and query-attention transformers
  # Inputs :  
  #   h_t of size      (bsz, 1, dim_emb)                            batch of input queries
  #   K_att of size    (bsz, nb_nodes+1, dim_emb*nb_layers_decoder) batch of query-attention keys for all decoding layers
  #   V_att of size    (bsz, nb_nodes+1, dim_emb*nb_layers_decoder) batch of query-attention values for all decoding layers
  #   mask of size     (bsz, nb_nodes+1)                            batch of masks of visited cities
  # Output :  
  #   prob_next_node of size (bsz, nb_nodes+1)
  classname = "Transformer decoder net",
  initialize = function(dim_emb, nb_heads, nb_layers_decoder){
    self$dim_emb = dim_emb
    self$nb_heads = nb_heads
    self$nb_layers_decoder = nb_layers_decoder
    self$decoder_layers = seq_len(nb_layers_decoder-1) |>
      lapply(\(x)decoder_layer(dim_emb, nb_heads)) |> 
      nn_module_list() |> 
      with_device(device=dvc)
    self$Wq_final = nn_linear(dim_emb, dim_emb) |> with_device(device=dvc)
  },
  reset_selfatt_keys_values = function(){
    for(l in seq_len(self$nb_layers_decoder-1)) self$decoder_layers[[l]]$reset_selfatt_keys_values()
  },
  forward = function(h_t, K_att, V_att, mask){
    # browser()
    for (l in seq_len(self$nb_layers_decoder)){
      K_att_l = K_att[,,((l-1)*self$dim_emb+1):(l*self$dim_emb)]#$contiguous()  # size(K_att_l)=(bsz, nb_nodes+1, dim_emb)
      V_att_l = V_att[,,((l-1)*self$dim_emb+1):(l*self$dim_emb)]#$contiguous()  # size(V_att_l)=(bsz, nb_nodes+1, dim_emb)
      # decoder layers with multiple heads (intermediate layers)
      if(l<self$nb_layers_decoder){h_t = self$decoder_layers[[l]](h_t, K_att_l, V_att_l, mask)}
      # decoder layers with single head (final layer)
      else{
        q_final = self$Wq_final(h_t)
        bsz = h_t$size(1)
        q_final = q_final$view(c(bsz, 1, self$dim_emb))
        attn_weights = my_MHA(q_final, K_att_l, V_att_l, 1, mask, 10)[[2]]
      }
    }
    attn_weights$squeeze(2) 
  }
)
# decoder(20, 2L, 3L)(torch_rand(3,1,20,device=dvc), torch_rand(3,16,60,device=dvc), torch_rand(3,16,60,device=dvc),torch_randint(0,2, c(3,16), dtype=torch_bool(),device=dvc))
# profvis::profvis(decoder(20, 2L, 3L)(torch_rand(3,1,20), torch_rand(3,16,60), torch_rand(3,16,60),torch_randint(0,2, c(3,16), dtype=torch_bool())))

# NN Positional encoding ----------------------------------------------------------------------------------------------------------------------------------
pos_enc <- function(d_model, max_len){
  # Create standard transformer PEs.
  # Inputs :  
  #   d_model is a scalar correspoding to the hidden dimension
  #   max_len is the maximum length of the sequence
  # Output :  
  #   pe of size (max_len, d_model), where d_model=dim_emb, max_len=1000
  # browser()
  pe = torch_zeros(max_len, d_model, device=dvc)
  position = torch_arange(1, max_len, dtype=torch_float(), device=dvc)$unsqueeze(2)
  div_term = torch_exp(torch_arange(1, d_model, 2, device=dvc)$float() * (-torch_log(torch_tensor(10000, device=dvc)) / d_model))
  pe[,1:d_model:2] = torch_sin(position * div_term)
  pe[,2:d_model:2] = torch_cos(position * div_term)
  pe
}
# pos_enc(20, 100) 
# pos_enc(20, 100) |> as_array() |> apply(2, \(x)round(x, 3)) |> as.data.frame()|> emphatic::hl(ggplot2::scale_color_viridis_c())


# NN TSP net ----------------------------------------------------------------------------------------------------------------------------------------------
TSP_net <- nn_module(
  # The TSP network is composed of two steps :
  #   Step 1. Encoder step : Take a set of 2D points representing a fully connected graph 
  #          and encode the set with self-transformer.
  #   Step 2. Decoder step : Build the TSP tour recursively/autoregressively, 
  #          i.e. one node at a time, with a self-transformer and query-transformer. 
  # Inputs : 
  #   x of size (bsz, nb_nodes, dim_emb) Euclidian coordinates of the nodes/cities
  # deterministic is a boolean : If True the salesman will chose the city with highest probability. 
  # If False the salesman will chose the city with Bernouilli sampling.
  # Outputs : 
  #   tours of size (bsz, nb_nodes) : batch of tours, i.e. sequences of ordered cities 
  # tours[b,t] contains the idx of the city visited at step t in batch b
  # sumLogProbOfActions of size (bsz,) : batch of sum_t log prob( pi_t | pi_(t-1),...,pi_0 )
  classname = "TSP net", 
  initialize = function(dim_input_nodes,dim_emb,dim_ff,nb_layers_encoder,nb_layers_decoder,nb_heads,max_len_PE,batchnorm=T){
    self$dim_emb = dim_emb
    
    # input embedding layer
    self$input_emb = nn_linear(dim_input_nodes, dim_emb) |> with_device(device=dvc)
    
    # encoder layer
    self$encoder = encoder(nb_layers_encoder, dim_emb, nb_heads, dim_ff, batchnorm)
    
    # vector to start decoding 
    self$start_placeholder = nn_parameter(torch_randn(dim_emb)) |> with_device(device=dvc)
    
    # decoder layer
    self$decoder = decoder(dim_emb, nb_heads, nb_layers_decoder) 
    self$WK_att_decoder = nn_linear(dim_emb, nb_layers_decoder* dim_emb) |> with_device(device=dvc)
    self$WV_att_decoder = nn_linear(dim_emb, nb_layers_decoder* dim_emb)  |> with_device(device=dvc)
    self$PE = pos_enc(dim_emb, max_len_PE) 
  },
  forward = function(x, deterministic=FALSE){
    # browser()
    # some parameters
    bsz = x$shape[1]
    nb_nodes = x$shape[2]
    # zero_to_bsz = torch_arange(bsz, device=x$device) # [1,...,bsz]
    
    # input embedding layer
    h = self$input_emb(x) # size(h)=(bsz, nb_nodes, dim_emb)
    dim_emb = h$size(3)
    
    # concat the nodes and the input placeholder that starts the decoding
    h = c(h, self$start_placeholder$expand(c(bsz, 1, -1))) |> torch_cat(dim=2) # size(start_placeholder)=(bsz, nb_nodes+1, dim_emb)
    
    # encoder layer
    h_encoder = self$encoder(h)[[1]] # size(h)=(bsz, nb_nodes+1, dim_emb)
    
    # list that will contain Long tensors of shape (bsz,) that gives the idx of the cities chosen at time t
    tours = NULL
    
    # list that will contain Float tensors of shape (bsz,) that gives the neg log probs of the choices made at time t
    sumLogProbOfActions = NULL
    
    # key and value for decoder    
    K_att_decoder = self$WK_att_decoder(h_encoder) # size(K_att)=(bsz, nb_nodes+1, dim_emb*nb_layers_decoder)
    V_att_decoder = self$WV_att_decoder(h_encoder) # size(V_att)=(bsz, nb_nodes+1, dim_emb*nb_layers_decoder)
    
    # input placeholder that starts the decoding
    # self$PE = self$PE$to(device = x$device) # it's already instantiated with correct device
    
    h_start = h_encoder[, nb_nodes, ] + self$PE[1]$expand(c(bsz,-1)) # size(h_start)=(bsz, dim_emb)
    
    # initialize mask of visited cities
    mask_visited_nodes = torch_zeros(bsz, nb_nodes+1, device=dvc)$bool() # False
    mask_visited_nodes[, nb_nodes + 1] = TRUE
    
    # clear key and val stored in the decoder
    self$decoder$reset_selfatt_keys_values()
    
    # construct tour recursively
    h_t = h_start
    for(t in seq_len(nb_nodes)){
      # compute probability over the next node in the tour
      prob_next_node = self$decoder(h_t, K_att_decoder, V_att_decoder, mask_visited_nodes) # size(prob_next_node)=(bsz, nb_nodes+1)
      
      # choose node with highest probability or sample with Bernouilli 
      if(deterministic){idx = torch_argmax(prob_next_node, dim=2)} # size(query)=(bsz,)
      else{idx = distr_categorical(prob_next_node)$sample()}# size(query)=(bsz,)
      # browser()
      # compute logprobs of the action items in the list sumLogProbOfActions   
      ProbOfChoices = prob_next_node$gather(dim=2, idx$unsqueeze(2))[,-1] # size(query)=(bsz,)
       
      sumLogProbOfActions <- c(sumLogProbOfActions, torch_log(ProbOfChoices + 1e-10))  # size(query)=(bsz,)
      
      # update embedding of the current visited node
      h_t = h_encoder$gather(dim=2, idx$unsqueeze(2)$expand(c(-1,dim_emb))$unsqueeze(2))$squeeze(2) # size(h_start)=(bsz, dim_emb)
      h_t = h_t + self$PE[t+1]$expand(c(bsz, -1))
      
      # update tour
      tours <- c(tours, idx)
      
      # update masks with visited nodes
      mask_visited_nodes = mask_visited_nodes$clone()
      mask_visited_nodes = mask_visited_nodes$scatter(dim=2, idx$unsqueeze(2), src = TRUE) 
    }
    # logprob_of_choices = sum_t log prob( pi_t | pi_(t-1),...,pi_0 )
    sumLogProbOfActions = torch_stack(sumLogProbOfActions, dim=2)$sum(dim=2) # size(sumLogProbOfActions)=(bsz,)
    
    # convert the list of nodes into a tensor of shape (bsz,num_cities)
    tours = torch_stack(tours, dim=2) # size(col_index)=(bsz, nb_nodes)
    
    list(tour_train=tours, sumLogProbOfActions=sumLogProbOfActions)
  }
)
# TSP_net(2L, 20L, 4L, 2L, 2L, 2L, 1000)(torch_rand(3, 15, 2, device=dvc), TRUE)

# profvis::profvis(TSP_net(2L, 20L, 4L, 2L, 2L, 2L, 1000)(torch_rand(3, 15, 2), TRUE))

# Train ---------------------------------------------------------------------------------------------------------------------------------------------------
test_tasks <- lapply(2021:2030, \(x)generate_task(args$nb_nodes, x) |> torch_tensor(device=dvc)) |> 
  torch_stack(dim=1)

tsts <- list.files("test_results/", full.names = T) |> 
  lapply(readRDS) |> 
  purrr::list_rbind() |> 
  dplyr::mutate(task_size = lapply(route, \(x)ifelse(length(x)%%2==0, length(x), length(x)-1)) |> unlist()) |>
  dplyr::filter(model=="Concorde" & task_size == args$nb_nodes) |> 
  dplyr::summarise(distance=mean(distance)) |> 
  dplyr::pull(distance)

train_net <- function(args, logs=TRUE){
  model_train = TSP_net(args$dim_input_nodes, args$dim_emb, args$dim_ff, 
                        args$nb_layers_encoder, args$nb_layers_decoder, args$nb_heads, args$max_len_PE,
                        batchnorm=args$batchnorm)
  
  model_baseline = TSP_net(args$dim_input_nodes, args$dim_emb, args$dim_ff, 
                           args$nb_layers_encoder, args$nb_layers_decoder, args$nb_heads, args$max_len_PE,
                           batchnorm=args$batchnorm)
  
  optimizer = optim_adamw(model_train$parameters, lr = args$lr) 

  # Prepare containers to store training loss and rewards for each epoch
  total_epochs = args$nb_epochs*args$nb_batch_per_epoch
  train_loss = vector(mode = "double", length = total_epochs)
  train_tour = vector(mode = "double", length = total_epochs)
  best_route = Inf
  best_dist = Inf
  
  model_train = model_train$to(device=dvc)
  model_baseline = model_baseline$to(device=dvc)
  model_baseline$eval()
  
  for (epoch in seq_len(args$nb_epochs)){
    start_time = Sys.time()
    model_train$train() 
    
    for(step in seq_len(args$nb_batch_per_epoch)){
      # generate a batch of random TSP instances    
      x = 100*torch_rand(args$bsz, args$nb_nodes, args$dim_input_nodes, device=dvc) # size(x)=(bsz, nb_nodes, dim_input_nodes) 
      
      # compute tours for model
      # c(tour_train, sumLogProbOfActions) %<-% model_train(x, deterministic=FALSE) 
      trn = model_train(x, deterministic=FALSE) # size(tour_train)=(bsz, nb_nodes), size(sumLogProbOfActions)=(bsz)
      
      with_no_grad({
        # compute tours for baseline
        tour_baseline = model_baseline(x, deterministic=TRUE)[[1]]
        
        # get the lengths of the tours
        L_train = calc_dist_batch(x, trn$tour_train) # size(L_train)=(bsz)
        L_baseline = calc_dist_batch(x, tour_baseline) # size(L_baseline)=(bsz)
      })
      
      # backprop
      loss = torch_mean((L_train - L_baseline)* trn$sumLogProbOfActions)
      optimizer$zero_grad()
      loss$backward()
      optimizer$step()

      # Store the loss and reward for this epoch
      loss <- loss$item()
      reward <- L_train$mean()$item()
      train_loss[(epoch-1)*args$nb_batch_per_epoch + step] <- loss
      train_tour[(epoch-1)*args$nb_batch_per_epoch + step] <- reward
      
      if (step %% 10 == 0 & logs) cat("SubEpoch: ", paste0(epoch, "_", step), "| Loss: ", loss, "| Reward: ", reward, "\n")
    }
    time_one_epoch = Sys.time()-start_time
    
    # Evaluate train model and baseline on 10k random TSP instances
    model_train$eval()
    mean_tour_length_train = 0
    mean_tour_length_baseline = 0
    for (step in seq_len(args$nb_batch_eval)){
      
      # generate a batch of random tsp instances   
      x = 100*torch_rand(args$bsz, args$nb_nodes, args$dim_input_nodes, device=dvc) 
      
      # compute tour for model and baseline
      with_no_grad({
        tour_train = model_train(x, deterministic=TRUE)[[1]]
        tour_baseline = model_baseline(x, deterministic=TRUE)[[1]]
        
        # get the lengths of the tours
        L_train = calc_dist_batch(x, tour_train)
        L_baseline = calc_dist_batch(x, tour_baseline) 
      })
      # L_tr and L_bl are tensors of shape (bsz,). Compute the mean tour length
      mean_tour_length_train = mean_tour_length_train + L_train$mean()$item()
      mean_tour_length_baseline = mean_tour_length_baseline + L_baseline$mean()$item()
    }
    mean_tour_length_train = mean_tour_length_train/ args$nb_batch_eval
    mean_tour_length_baseline = mean_tour_length_baseline/ args$nb_batch_eval
    
    # Evaluate train model and baseline and update if train model is better
    update_baseline = mean_tour_length_train + args$tol < mean_tour_length_baseline
    if(update_baseline) model_baseline$load_state_dict(model_train$state_dict())
    
    # Evaluate model against test tours used across AI optimisation project https://github.com/dkibalnikov/ai_optimization
    if(args$nb_nodes==16){
      tour_test = model_train(test_tasks, deterministic=TRUE)[[1]]
      mean_tour_length_test = calc_dist_batch(test_tasks, tour_test)$mean()$item()
      gap = mean_tour_length_test/tsts - 1
      if(mean_tour_length_test<best_dist){
        best_route=as_array(tour_test[1]) # update best solution
        best_dist=mean_tour_length_test # update best solution distance
      } 
    }else{
      best_route <- NULL
      tour_test <- torch_zeros(1)
      mean_tour_length_test <- "-"
      gap <- NULL
    }
    # Print and save in txt file
    if(logs)cat("Epoch: ", epoch, 
        "| Time elapsed: ", format(round(Sys.time() - start_time, 2)), 
        "| L_train: ", mean_tour_length_train, 
        "| L_base: ", mean_tour_length_baseline,
        "| L_test: ", mean_tour_length_test, 
        "| Update baseline: ", update_baseline, 
        "| Gap: ", scales::percent(gap), "\n")
  }
  # Return a list containing the training results
  list(loss = train_loss, reward = train_tour, best = best_route, last = as_array(tour_test[1]), mdl = model_baseline)
}

res <- train_net(args=args)

# Evaluate NN ---------------------------------------------------------------------------------------------------------------------------------------------

cities <- generate_task()
dist_mtrx <- dist(cities) |> as.matrix()

# Best solution 
p1 <- prep4plot(cities, res$best) |> 
  plot_tour() + 
  labs(title = paste0("Лучшее решение: ", calc_dist4mtrx(dist_mtrx, res$best)|>round(2)), 
       col = "Маршрут", x = "x", y = "y")

# Last solution 
p2 <- prep4plot(cities, res$last) |> 
  plot_tour() +
  labs(title = paste0("Итоговое решение: ", calc_dist4mtrx(dist_mtrx, res$last)|>round(2)), 
       col = "Маршрут", x = "x", y = "y")

# Training perfomance metrics 
p3 <- tibble::tibble(n = seq_along(res$loss), loss = res$loss, reward = res$reward) |>
  tidyr::pivot_longer(-1, names_to = "variable") |>
  ggplot(aes(n, value, col = variable)) + 
  geom_line(alpha = .3) + 
  geom_vline(data = tibble::tibble(ep=seq_len(args$nb_epochs)*args$nb_batch_per_epoch), 
             aes(xintercept = ep), lty=5) + 
  geom_smooth(formula = y ~ s(x, bs = "cs"), method = 'gam') +
  facet_wrap(~variable, nrow = 2, scales = "free") +
  geom_point(data = ~dplyr::filter(., n==which.min(res$reward)), col = "firebrick") + 
  labs(title = "Характеристики обучения нейронной сети", col = "", y = "Значение", x = "Итерация обучения")

# Combine plots to one 
(p1+p2)/p3 + plot_annotation("TSP Transformers")

# Check generalization capability 
new_task <- generate_task(seed = 2023)  
new_task_tnsr <- torch_tensor(new_task)$unsqueeze(1)$expand(c(2,16,2)) # batch should not contain only 1 element
new_task_sol <- res$mdl(new_task_tnsr, deterministic = TRUE)$tour_train[1,] |> as_array()
new_task_dist_mtrx <- dist(new_task) |> as.matrix()

new_task_sol_dist <- -calc_dist4mtrx(new_task_dist_mtrx, new_task_sol) |> round(2)

prep4plot(new_task, new_task_sol) |> 
  plot_tour() +
  labs(title = paste0("Итоговое решение: ", -new_task_sol_dist), 
       col = "Маршрут", x = "x", y = "y")

# Save results ------------------------------------------------------------
res$mdl$eval()
saveRDS(res, "test_samples/transformers.rds")
torch_save(res$mdl, "test_samples/transformers_baseline")
torch_save(res$mdl$PE, "test_samples/transformers_PE")

# Save operation breaks dependencies for Positional Encoding, so save/load individually 
res <- readRDS("test_samples/transformers_16.rds")
res$mdl <- torch_load("test_samples/transformers_baseline_16")
res$mdl$PE <- torch_load("test_samples/transformers_PE_16")
res$mdl(new_task_tnsr, deterministic = TRUE) # test loaded model 

# Wrap model and calculate batch ------------------------------------------

get_transformers <- function(task, n_cities = 16){
  
  task_tnsr <- torch_tensor(task)$unsqueeze(1)$expand(c(2, n_cities,2)) # batch should not contain only 1 element
  task_dist_mtrx <- dist(task) |> as.matrix()
  
  start_time = Sys.time() 
  task_sol <- res$mdl(task_tnsr, deterministic = TRUE)$tour_train[1,] |> as_array()
  duration <- Sys.time() - start_time
  
  task_sol_dist <- calc_dist4mtrx(task_dist_mtrx, task_sol) |> round(2)
  
  tibble::tibble(model = "Transformers", duration = duration, 
                 distance = task_sol_dist, route = list(task_sol))
}

get_transformers(new_task)

res_16 <- calc_tours(get_transformers, n_cities = 16, runs = 1)

args$nb_nodes = 32L # TSP32
res <- train_net(args=args)
res_32 <- calc_tours(get_transformers, n_cities = 32, runs = 1) 
res$mdl$eval()
saveRDS(res, "test_samples/transformers_32.rds")
torch_save(res$mdl, "test_samples/transformers_baseline_32")
torch_save(res$mdl$PE, "test_samples/transformers_PE_32")

args$nb_nodes = 64L # TSP64
res <- train_net(args=args)
res_64 <- calc_tours(get_transformers, n_cities = 64, runs = 10) 

saveRDS(res_16, "test_results/transformers_16nodes.rds")
saveRDS(res_32, "test_results/transformers_32nodes.rds")
saveRDS(res_64, "test_results/transformers_64nodes.rds")