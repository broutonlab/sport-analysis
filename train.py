import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.constants import BATCH_SIZE, EPOCHS, BEST_LOSS, device
from src.data.load_data import load_paths
from src.data.make_dataset import KeypointDataset
from src.models.model import MobileNetV1, MOBILENET_V1_CHECKPOINTS, load_variables
from src.models.predict import train, validate
from src.visualisation.visualize import visualize_model

paths_list_train, paths_list_val = load_paths("./data/raw")

samples_train = list(paths_list_train)
samples_val = list(paths_list_val)

print("samples_train", len(samples_train))
print("samples_val", len(samples_val))

# create dataset
print("\n-------------- PREPARING DATA --------------\n")
train_data = KeypointDataset(samples_train)
val_data = KeypointDataset(samples_val)
print("\n-------------- DATA PREPRATION DONE --------------\n")
# prepare data loaders
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

# model
model_version = 50

model = MobileNetV1(model_version)
# load weights
checkpoint = MOBILENET_V1_CHECKPOINTS[model_version]
state_dict = load_variables(checkpoint)
model.load_state_dict(state_dict)

model.to(device)
# change the last 2 layers
# 4+1 = классификатор наличия точки в блоке,
# 4*2 = количество дельта, которые надо расчитать
model.heatmap = nn.Conv2d(model.last_depth, 27, 2, 2).double().to(device)
model.offset = nn.Conv2d(model.last_depth, 52, 2, 2).double().to(device)

# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
# loss for regression(Sx, Sy)
criterion = nn.MSELoss()
# loss for classification
criterion_point_class = torch.nn.BCELoss()

visualize_model(samples_train, 3, train_data, model)

train_loss = []
val_loss = []
train_acc = []
val_acc = []

writer = SummaryWriter()

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1} of {EPOCHS}")
    val_best_loss = BEST_LOSS
    train_epoch_accuracy, train_epoch_loss = train(
        model, train_loader, train_data, criterion, optimizer, criterion_point_class
    )
    val_epoch_accuracy, val_epoch_loss, best_model_wts = validate(
        model, val_loader, val_data, criterion, criterion_point_class, val_best_loss
    )

    train_acc.append(train_epoch_accuracy)
    val_acc.append(val_epoch_accuracy)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)

    if epoch % 5 == 0:
        model.load_state_dict(best_model_wts)
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"./checkpoints/field_keypoints_best_{epoch}.pd")
    print(f"Train Acc: {train_epoch_accuracy:.4f}")
    writer.add_scalar("Train Acc", train_epoch_accuracy, epoch)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    writer.add_scalar("Train Loss", train_epoch_loss, epoch)
    print(f"Val Acc: {val_epoch_accuracy:.4f}")
    writer.add_scalar("Val Acc", val_epoch_accuracy, epoch)
    print(f"Val Loss: {val_epoch_loss:.4f}")
    writer.add_scalar("Val Loss", val_epoch_loss, epoch)

# load best model weights
model.load_state_dict(best_model_wts)

torch.save(model.state_dict(), "models/exp0/field_keypoints_best.pd")
# visualize results
visualize_model(samples_val, 5, val_data, model)
visualize_model(samples_train, 5, train_data, model)
