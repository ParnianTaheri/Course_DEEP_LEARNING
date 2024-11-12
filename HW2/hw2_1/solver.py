# You are not allowed to import any other libraries or modules.

import torch
import torch.nn as nn
import numpy as np


def train(model, criterion, optimizer, train_dataloader, num_epoch, device):
    model.to(device)
    avg_train_loss, avg_train_acc = [], []

    for epoch in range(num_epoch):
        model.train()
        batch_train_loss, batch_train_acc = train_one_epoch(model, criterion, optimizer, train_dataloader, device)
        avg_train_acc.append(np.mean(batch_train_acc))
        avg_train_loss.append(np.mean(batch_train_loss))

        print(f'\nEpoch [{epoch}] Average training loss: {avg_train_loss[-1]:.4f}, '
              f'Average training accuracy: {avg_train_acc[-1]:.4f}')

    return model


def train_one_epoch(model, criterion, optimizer, train_dataloader, device):
    batch_train_loss = []
    batch_train_acc = []

    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        labels = labels.argmax(dim=1)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == labels).sum().item() / labels.size(0)*100
        
        batch_train_loss.append(loss.item())
        batch_train_acc.append(accuracy)

    return batch_train_loss, batch_train_acc



def test(model, test_dataloader, device):
    model.to(device)
    model.eval()
    batch_test_acc = []

    #TODO: Test the model on the given test dataset and store accuracy in batch_test_acc. This function return nothing.
    # Remember you should disable gradient computation during testing.

    with torch.no_grad():  
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            outputs = model(inputs)

            # Calculate accuracy
            labels = labels.argmax(dim=1)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == labels).sum().item() / labels.size(0)*100
            batch_test_acc.append(accuracy)

    print(f"The test accuracy is {torch.mean(torch.tensor(batch_test_acc)):.4f}.\n")
