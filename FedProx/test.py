import numpy as np
import torch
from torch.utils.data import DataLoader


def testing(model, dataset, bs, criterion, num_classes):
    # test loss
    test_loss = 0.0
    correct_class = list(0. for i in range(num_classes))
    total_class = list(0. for i in range(num_classes))

    test_loader = DataLoader(dataset, batch_size=bs)
    l = len(test_loader)
    model.eval()
    for data, labels in test_loader:

        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()

        output = model(data)
        loss = criterion(output, labels)
        test_loss += loss.item() * data.size(0)

        _, pred = torch.max(output, 1)

        correct_tensor = pred.eq(labels.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(
            correct_tensor.cpu().numpy())

        # test accuracy for each object class
        for i in range(num_classes):
            label = labels.data[i]
            correct_class[label] += correct[i].item()
            total_class[label] += 1

    # avg test loss
    test_loss = test_loss / len(test_loader.dataset)
    # print("Test Loss: {:.6f}\n".format(test_loss))

    # print test accuracy
    # for i in range(10):
    #   if total_class[i]>0:
    #     print('Test Accuracy of %5s: %2d%% (%2d/%2d)' %
    #           (classes[i], 100 * correct_class[i] / total_class[i],
    #           np.sum(correct_class[i]), np.sum(total_class[i])))
    #   else:
    #     print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    test_accuracy = 100. * np.sum(correct_class) / np.sum(total_class)

    # print('\nFinal Test  Accuracy: {:.3f} ({}/{})'.format(
    #       100. * np.sum(correct_class) / np.sum(total_class),
    #       np.sum(correct_class), np.sum(total_class)))
    return test_loss, test_accuracy