import os
import torch
from utils.trainer import Trainer
from utils.data import get_args, get_vocabs
from utils.utils import set_seed, get_model
from torch.utils.tensorboard import SummaryWriter


args = get_args()
set_seed(args.seed)


vocabs = get_vocabs(os.path.join(args.path, 'data', 'vocabs'))
model = get_model(args.model, vocabs, args.hidden_dim, args.main_view, args.aux_views).to('cuda')
all_p = list(model.parameters())
table = list(model.EmbeddingTable.parameters())
other = [p for p in model.parameters() if p not in set(table)]
optimizer = torch.optim.Adam([{'params': table, 'lr': args.lr * 10}, {'params': other, 'lr': args.lr}])


total_steps = 4000000 * 21 // args.batch_size
warmup_steps = int(total_steps * 0.05)
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.0001, end_factor=1.0, total_iters=warmup_steps)
constant_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=total_steps - warmup_steps)
lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, constant_scheduler], milestones=[warmup_steps])


aux_views_str = '_'.join(args.aux_views)
cur_name = f"{args.model}_{args.main_view}_{'none' if len(aux_views_str) == 0 else aux_views_str}_" + str(args.lr) + "_" + str(args.lamda)
writer = SummaryWriter(os.path.join(args.path, 'results', 'runs', cur_name))


trainer =  Trainer(args, model, optimizer, lr_scheduler, writer)
trainer.run()
