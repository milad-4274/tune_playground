# 1. Wrap your PyTorch model in an objective function.
import torch
from ray import tune, air
from ray.air import session
from ray.tune.search.optuna import OptunaSearch
from get_data import gen_data
from model import ClassificationModel, train
import numpy as np



DEVICE = "cpu"



# 1. Wrap a PyTorch model in an objective function.
def objective(config):
    data = gen_data()
    model = ClassificationModel()

    model.to(DEVICE)

    optimizer = torch.optim.SGD(  # Tune the optimizer
        model.parameters(), lr=config["lr"], momentum=config["momentum"]
        # model.parameters(), lr=config["lr"]
    )

    loss_fn = torch.nn.BCEWithLogitsLoss()

    loss, accuracy = train(model, optimizer, data, loss_fn)  # Train the model
    # acc = test(model, data, loss_fn)  # Compute test accuracy
    session.report({"last_accuracy": accuracy, "last_loss": loss})  # Report to Tune
    # session.report({"last_accuracy": map(int,accuracies["test"]), "last_loss": losses["test"] })  # Report to Tune

    return {"last_accuray":accuracy,"last_loss": loss}

# 2. Define a search space and initialize the search algorithm.
search_space = {"lr": tune.loguniform(1e-2, 1e-1), 
                "momentum": tune.uniform(0.1, 0.9)
                }
# search_space = {
#     "lr": tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
#     "momentum": tune.uniform(0.1, 0.9),
# }

# search_space = {
#     "lr": tune.grid_search([0.001, 0.01, 0.1, 1.0]),
#     "momentum": tune.choice([.1, .2, .3, .4]),
# }



tuner = tune.Tuner(
    objective,
    param_space=search_space,
    tune_config=tune.TuneConfig(
        num_samples=5,
        # scheduler=hyperband,
    ),
)
results = tuner.fit()

dfs = {result.log_dir: result.metrics_dataframe for result in results}
print("1      ",results.get_best_result(metric="last_accuracy", mode="max").config)
print("2      ",results.get_best_result(metric="last_loss", mode="min").config)