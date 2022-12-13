from torch import nn
import torch



class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2,10)
        self.layer2 = nn.Linear(10,10)
        self.layer3 = nn.Linear(10,1)
        self.relu = nn.ReLU()

    def forward(self ,x):
        # where should we put our non-linear activation function
        return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))


# Calculate accuracy
def calc_accuracy(output, predictions):
    assert len(output) == len(predictions) 
    correct = torch.eq(output, predictions).sum().item()
    acc = (correct / len(predictions)) * 100
    
    return acc




def train(model, optimizer, data, loss_fn):

    epochs = 3000
    losses = {x: [] for x in ["train","test"]}
    accuracies = {x: [] for x in ["train", "test"]} 
    X_train, X_test, y_train, y_test = data

    for epoch in range(epochs):

        model.train()
        y_logits = model(X_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))

        train_loss = loss_fn(y_logits, y_train)
        train_acc = calc_accuracy(y_pred, y_train)

        losses["train"].append(train_loss.detach().cpu().numpy())
        accuracies["train"].append(train_acc)

        optimizer.zero_grad()

        train_loss.backward()

        optimizer.step()

        model.eval()
        with torch.inference_mode():
            test_logits = model(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))

            test_loss = loss_fn(test_logits,
                                y_test)
            test_acc = calc_accuracy(test_pred, y_test)


            losses["test"].append(test_loss.item())
            accuracies["test"].append(test_acc)


        if epoch == epochs - 1:
            print(f"Epoch {epoch} / {epochs} | train_loss: {train_loss:.5f} | train_acc: {train_acc:.2f} | test_loss: {test_loss:.5f} | test_acc: {test_acc:.2f}")


        # model.train()
        # output_logits = model(X_train).squeeze()
        # output_probs = torch.sigmoid(output_logits)
        # output_labels = torch.round(output_probs)
        # train_loss = loss_fn(output_logits, y_train)
        # train_acc = calc_accuracy(output_labels, y_train)

        # losses["train"].append(train_loss.detach().cpu().numpy())
        # accuracies["train"].append(train_acc)

        # optimizer.zero_grad()
        # train_loss.backward()
        # optimizer.step()

        # model.eval()
        # with torch.inference_mode():
        #     test_logits = model(X_test).squeeze()
        #     test_probs = torch.sigmoid(test_logits)
        #     test_labels = torch.round(test_probs)
        #     test_loss = loss_fn(test_logits, y_test)
        #     test_acc=  calc_accuracy(test_labels, y_test)

        #     losses["test"].append(test_loss.detach().cpu().numpy())
        #     accuracies["test"].append(train_acc)
        
        # print(f"Epoch {epoch}/{epochs} | train_loss: {train_loss} | train_acc: {train_acc} | test_loss: {test_loss} | test_acc: {test_acc}")
    return losses["test"][-1], accuracies["test"][-1]




