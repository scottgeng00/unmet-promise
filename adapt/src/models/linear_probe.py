import torch

from src.models.zeroshot import get_zeroshot_classifier
from src.models.modeling import ClassificationHead
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models.utils import cosine_lr
from tqdm import tqdm

DEFAULT_HPARAMS = {
    'lr': 1e-2,
    'bsz': 256,
    'epochs': 200,
    'wd': 0.1,
    'warmup_length': 200,
}

def linear_probe(args, image_enc, dataset, **kwargs):
    hparams = DEFAULT_HPARAMS.copy()
    hparams.update(kwargs)

    ############################## Setup Params ##############################
    lr = hparams['lr']
    bsz = hparams['bsz']
    epochs = hparams['epochs']
    wd = hparams['wd']
    warmup_length = hparams['warmup_length']

    input_key = 'features'
    
    original_batch_size = args.batch_size
    args.batch_size = bsz

    ############################## Setup Model ##############################
    model = ClassificationHead(normalize=True, weights=None, num_classes=len(dataset.classnames))
    model = model.cuda()
    devices = list(range(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model, device_ids=devices)
    print("Intiailizing LP on", devices)

    ########################## Setup Loss and Optimizer #######################
    loss_fn = torch.nn.CrossEntropyLoss()

    num_batches = len(dataset.train_loader)
    param_groups = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=wd, betas=(0.9, 0.95))
    params = [p for p in model.parameters() if p.requires_grad]
    scheduler = cosine_lr(optimizer, lr, warmup_length, epochs * num_batches)
    
    ############################## Train ##############################
    for epoch in tqdm(range(epochs), desc="LP Training Epochs"):
        data_loader = get_dataloader(dataset, is_train=True, args=args, image_encoder=image_enc)
        model.train()
        for i, batch in enumerate(data_loader):
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()
            
            batch = maybe_dictionarize(batch)
            inputs = batch[input_key].cuda()
            labels = batch['labels'].cuda()
            
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

    ############################## Cleanup ##############################
    args.batch_size = original_batch_size

    return model.module
