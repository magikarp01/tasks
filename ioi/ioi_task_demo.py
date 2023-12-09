#%%
from tasks.ioi.IOITask import IOIDataset
import torch as t
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

#%%

model = HookedTransformer.from_pretrained(
    'NeelNanda/GELU_2L512W_C4_Code',
    center_writing_weights=False,
    center_unembed=False,
    fold_ln=False,
    device=device,
)
model.set_use_hook_mlp_in(True)
model.set_use_split_qkv_input(True)
model.set_use_attn_result(True)

# %%
N=3
task = IOIDataset(
    model,
    prompt_type='mixed',
    N=N,
    tokenizer=model.tokenizer,
    prepend_bos=False,
    seed=1,
    device=device
)
# %%
task_loader = DataLoader(task, batch_size=2)

# %%
data = next(iter(task_loader))
clean_logits = model(data['clean_tok'])
corr_logits = model(data['corr_tok'])

losses = IOIDataset.get_loss(clean_logits, corr_logits, data)

# Losses is of type (clean_loss, corr_loss)
# Each loss is mean logit diff (logit of S - logit of IO)
# Smaller loss is better obv

print(f'Clean loss: {losses[0]}, Corr loss: {losses[1]}')