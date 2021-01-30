import time

import numpy as np
import torch
import torch.nn as nn
import copy


from GenerateLocalEpochs import GenerateLocalEpochs
from local_train import ClientUpdate
from test import testing


def training(model, rounds, batch_size, lr, ds, data_dict, test_data_dict, C, K, E, mu, percentage, plt_title, plt_color, target_test_accuracy, algorithm="fedprox"):

    """
    Function implements the Federated Averaging Algorithm from the FedAvg paper.
    Specifically, this function is used for the server side training and weight update

    Params:
      - model:           PyTorch model to train
      - rounds:          Number of communication rounds for the client update
      - batch_size:      Batch size for client update training
      - lr:              Learning rate used for client update training
      - ds:              Dataset used for training
      - data_dict:       Type of data partition used for training (IID or non-IID)
      - test_data_dict:  Data used for testing the model
      - C:               Fraction of clients randomly chosen to perform computation on each round
      - K:               Total number of clients
      - E:               Number of training passes each client makes over its local dataset per round
      - mu:              proximal term constant
      - percentage:      percentage of selected client to have fewer than E epochs
    Returns:
      - model:           Trained model on the server
    """

    # global model weights
    global_weights = model.state_dict()

    # training loss
    list_loss_train = []        #选中客户端的损失平均
    list_loss_train_2 = []     #所有训练数据的损失平均
    list_acc_train = []

    # test accuracy
    list_loss_test = []
    list_acc_test = []


    # store last loss for convergence
    last_loss = 0.0

    # total time taken
    total_time = 0

    print(f"System heterogeneity set to {percentage}% stragglers.\n")
    print(f"Picking {max(int(C * K), 1)} random clients per round.\n")

    start = time.time()
    for curr_round in range(0, rounds):
        w, local_loss, lst_local_train_time = [], [], []

        m = max(int(C * K), 1)

        heterogenous_epoch_list = GenerateLocalEpochs(percentage, size=m, max_epochs=E)
        heterogenous_epoch_list = np.array(heterogenous_epoch_list)

        S_t = np.random.choice(range(K), m, replace=False)
        S_t = np.array(S_t)

        # For Federated Averaging, drop all the clients that are stragglers
        if algorithm == 'fedavg':
            stragglers_indices = np.argwhere(heterogenous_epoch_list < E)
            heterogenous_epoch_list = np.delete(heterogenous_epoch_list, stragglers_indices)
            S_t = np.delete(S_t, stragglers_indices)

        for k, epoch in zip(S_t, heterogenous_epoch_list):
            local_update = ClientUpdate(dataset=ds, batchSize=batch_size, learning_rate=lr, epochs=epoch,
                                        idxs=data_dict[k], mu=mu, algorithm=algorithm)
            weights, loss, local_train_time = local_update.train(model=copy.deepcopy(model))

            w.append(copy.deepcopy(weights))
            local_loss.append(copy.deepcopy(loss))
            lst_local_train_time.append(local_train_time)

        # calculate time to update the global weights
        global_start_time = time.time()

        # updating the global weights
        weights_avg = copy.deepcopy(w[0])
        for k in weights_avg.keys():
            for i in range(1, len(w)):
                weights_avg[k] += w[i][k]

            weights_avg[k] = torch.div(weights_avg[k], len(w))

        global_weights = weights_avg

        global_end_time = time.time()

        # calculate total time
        total_time += (global_end_time - global_start_time) + sum(lst_local_train_time) / len(lst_local_train_time)

        # move the updated weights to our model state dict
        model.load_state_dict(global_weights)

        # loss
        loss_avg = sum(local_loss) / len(local_loss)

        # test accuracy
        criterion = nn.CrossEntropyLoss()
        test_loss, test_accuracy = testing(model, test_data_dict, 128, criterion, 10)
        train_loss, train_accuracy = testing(model, test_data_dict, 128, criterion, 10)

        print(
            f"Round: {curr_round}... \tAverage Train Loss: {round(loss_avg, 3)}... \tTest Loss: {test_loss}... \tTest Accuracy: {test_accuracy}")

        list_loss_train.append(loss_avg)
        list_loss_train_2.append(train_loss)
        list_acc_train.append(train_accuracy)

        list_acc_test.append(test_accuracy)
        list_loss_test.append(test_loss)


        # break if we achieve convergence, i.e., loss between two consecutive rounds is <0.0001
        if algorithm == 'fedprox' and abs(loss_avg - last_loss) < 0.0001:
            rounds = curr_round
            break

        # update the last loss
        last_loss = loss_avg
    end = time.time()

    print('list_loss_train:', list_loss_train)
    print('list_loss_test:', list_loss_test)
    print('list_acc_test:', list_acc_test)
    print('list_loss_train_2:', list_loss_train_2)
    print('list_acc_train:', list_acc_train)
    print('time', (end - start) / 60)

    print("Training Done!")
    # print("Total time taken to Train: {}".format(get_time_format(total_time)))

    return model