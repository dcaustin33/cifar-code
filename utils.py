import torch


def get_accuracy(predictions, labels):
    if len(predictions) > 0:
        _, predicted = torch.max(predictions, 1)
        acc1 = (predicted == labels).sum()

        _, pred = predictions.topk(5)
        labels = labels.unsqueeze(1).expand_as(pred)
        acc5 = (labels == pred).any(dim = 1).sum()
        return acc1, acc5
    return 0, 0