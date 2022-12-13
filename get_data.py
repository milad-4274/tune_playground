import sklearn
from sklearn.datasets import make_circles
import torch
from sklearn.model_selection import train_test_split



def gen_data(n_samples=1000):

    X,y = make_circles(n_samples,
                    noise = 0.03,
                    random_state=42)

    # convert data to tensor


    X = torch.from_numpy(X).type(torch.float)
    y = torch.from_numpy(y).type(torch.float)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

