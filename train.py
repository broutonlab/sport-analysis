import argparse
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.load_data import load_paths
from src.data.make_dataset import KeypointDataset
from src.models.model import prepare_model_to_train
from src.models.predict import train, validate
from src.options.base_options import BATCH_SIZE, BEST_LOSS, device, EPOCHS
from src.visualisation.visualize import visualize_model

"""-------------- GET ARGUMENTS --------------"""

parser = argparse.ArgumentParser(description=" ")
parser.add_argument(
    "--dataset",
    type=str,
    default="./data/raw",
    help="path to video. (default:./data/raw)",
)
parser.add_argument(
    "--exp_name",
    type=str,
    default="exp_last",
    help="path to video. (default:exp_last)",
)

args = parser.parse_args()

"""-------------- PREPARING DATA --------------"""

paths_list_train, paths_list_val = load_paths(args.dataset)

samples_train = list(paths_list_train)
samples_val = list(paths_list_val)

print("samples_train ", len(samples_train))
print("samples_val ", len(samples_val))


train_data = KeypointDataset(samples_train)
val_data = KeypointDataset(samples_val)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)


"""-------------- PREPARING MODEL AND PARAMS --------------"""

model = prepare_model_to_train(device=device)
# Visualize some data
visualize_model(samples_train, 1, train_data, model)
# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Loss for regression
criterion = nn.MSELoss()
# Loss for classification
criterion_point_class = torch.nn.BCELoss()
# For tensorboard statistics
writer = SummaryWriter()


if not os.path.isdir(os.path.join("./models/", args.exp_name)):
    os.mkdir(os.path.join("./models/", args.exp_name))
"""-------------- TRAINING PROCESS --------------"""

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1} of {EPOCHS}")
    val_best_loss = BEST_LOSS
    train_epoch_loss = train(
        model, train_loader, train_data, criterion, optimizer, criterion_point_class
    )
    val_epoch_loss, best_model_wts = validate(
        model, val_loader, val_data, criterion, criterion_point_class, val_best_loss
    )

    if epoch % 5 == 0:
        model.load_state_dict(best_model_wts)
    if epoch % 10 == 0:
        torch.save(
            model.state_dict(),
            os.path.join("./models/", args.exp_name, f"field_keypoint_best_{epoch}.pd"),
        )
    print(f"Train Loss: {train_epoch_loss:.4f}")
    writer.add_scalar("Train Loss", train_epoch_loss, epoch)
    print(f"Val Loss: {val_epoch_loss:.4f}")
    writer.add_scalar("Val Loss", val_epoch_loss, epoch)

# Load best model weights
model.load_state_dict(best_model_wts)
# Save best weights
torch.save(
    model.state_dict(),
    os.path.join("./models/", args.exp_name, "field_keypoint_best.pd"),
)
# Visualize results
visualize_model(samples_val, 5, val_data, model)
visualize_model(samples_train, 5, train_data, model)
