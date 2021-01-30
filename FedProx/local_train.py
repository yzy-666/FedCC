import copy
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class ClientUpdate(object):
    def __init__(self, dataset, batchSize, learning_rate, epochs, idxs, mu, algorithm):
        self.mu = mu
        self.train_loader = DataLoader(CustomDataset(dataset, idxs), batch_size=batchSize, shuffle=True)
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.epochs = epochs

    def train(self, model):
        # print("Client training for {} epochs.".format(self.epochs))
        criterion = nn.CrossEntropyLoss()
        proximal_criterion = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.5)

        # use the weights of global model for proximal term calculation
        global_model = copy.deepcopy(model)

        # calculate local training time
        start_time = time.time()

        e_loss = []
        for epoch in range(1, self.epochs + 1):

            train_loss = 0.0

            model.train()
            for data, labels in self.train_loader:

                if torch.cuda.is_available():
                    data, labels = data.cuda(), labels.cuda()

                # clear the gradients
                optimizer.zero_grad()
                # make a forward pass
                output = model(data)

                # calculate the loss + the proximal term
                _, pred = torch.max(output, 1)

                if self.algorithm == 'fedprox':
                    proximal_term = 0.0

                    # iterate through the current and global model parameters
                    for w, w_t in zip(model.parameters(), global_model.parameters()):
                        # update the proximal term
                        # proximal_term += torch.sum(torch.abs((w-w_t)**2))
                        proximal_term += (w - w_t).norm(2)

                    loss = criterion(output, labels) + (self.mu / 2) * proximal_term
                else:
                    loss = criterion(output, labels)

                # do a backwards pass
                loss.backward()
                # perform a single optimization step
                optimizer.step()
                # update training loss
                train_loss += loss.item() * data.size(0)

            # average losses
            train_loss = train_loss / len(self.train_loader.dataset)
            e_loss.append(train_loss)

        total_loss = sum(e_loss) / len(e_loss)

        return model.state_dict(), total_loss, (time.time() - start_time)