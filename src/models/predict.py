import copy

from tqdm import tqdm

from src.options.base_options import BEST_LOSS, device


def train(model, dataloader, data, criterion, optimizer, criterion_point_class):
    """."""
    print("Training")
    model.train()
    train_running_loss = 0.0
    counter = 0
    # Calculate the number of batches
    num_batches = int(len(data) / dataloader.batch_size)
    for _, data in tqdm(enumerate(dataloader), total=num_batches):
        counter += 1
        optimizer.zero_grad()
        images, real_headmap, real_offset = (
            data["image"].to(device),
            data["real_headmap"].to(device),
            data["real_offset"].to(device),
        )

        results = model(images)
        heatmaps, offsets = results
        loss_cross = criterion_point_class(heatmaps, real_headmap)
        loss_mse = criterion(offsets, real_offset)
        loss = loss_mse + loss_cross
        train_running_loss += loss.item()

        loss.backward()
        optimizer.step()

    train_loss = train_running_loss / counter
    return train_loss


def validate(
    model, dataloader, data, criterion, criterion_point_class, best_loss=BEST_LOSS
):
    """."""
    print("Validating")
    model.eval()
    val_running_loss = 0.0
    counter = 0
    # Calculate the number of batches
    num_batches = int(len(data) / dataloader.batch_size)
    for _, data in tqdm(enumerate(dataloader), total=num_batches):
        counter += 1
        images, real_headmap, real_offset = (
            data["image"].to(device),
            data["real_headmap"].to(device),
            data["real_offset"].to(device),
        )

        results = model(images)
        heatmaps, offsets = results

        loss_cross = criterion_point_class(heatmaps, real_headmap)
        loss_mse = criterion(offsets, real_offset)
        loss = loss_mse + loss_cross
        val_running_loss += loss.item()

    # Copy the best model
    if val_running_loss < best_loss:
        best_loss -= best_loss - val_running_loss
        best_model_wts = copy.deepcopy(model.state_dict())

    if counter == 0:
        val_loss = 0
    else:
        val_loss = val_running_loss / counter
    return val_loss, best_model_wts
