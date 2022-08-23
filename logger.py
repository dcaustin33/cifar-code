import wandb

def log_metrics(metrics: dict, 
                step: int
                wandb = None, ) -> None:
    for i in metrics:
        if i in ['total', 'Loss', 'LR'] or i[:3] == 'Val': continue
        if 'Accuracy' in i:
            metrics[i] = metrics[i] / metrics['total']

    print(step, "Loss:", round(metrics['Loss'].item(), 2), 'Acc', round(metrics['ccuracy'].item(), 2), metrics['LR'] )

    if wandb:
        wandb.log(metrics, step = step)

    metrics = {}
    metrics['total'] = 0
    metrics['Loss'] = 0
    metrics['Accuracy'] = 0
    metrics['Accuracy Top 5'] = 0
        
    return