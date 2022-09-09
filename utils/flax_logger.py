import wandb

def log_metrics(metrics: dict, 
                step: int,
                wandb = None, 
               train = True) -> None:
    for i in metrics:
        if i in ['total', 'Loss', 'LR'] or i[:3] == 'Val': continue
        if 'Accuracy' in i:
            metrics[i] = metrics[i] / metrics['total']

    if train:
        print('Train', step, "Loss:", round(metrics['Loss'], 2), 'Acc', round(metrics['Accuracy'], 2))
    '''else:
        print('Val', step, 'Acc', round(metrics['Accuracy'], 2))'''
    
    if not train:
        new_metrics = {}
        for i in metrics:
            new_metrics['Val ' + i] = metrics[i]
        metrics = new_metrics
    if wandb:
        wandb.log(metrics, step = step)

    metrics = {}
    metrics['total'] = 0
    metrics['Loss'] = 0
    metrics['Accuracy'] = 0
    metrics['Accuracy Top 5'] = 0
        
    return metrics