import copy

from tqdm import tqdm

from src.constants import NUM_KEYPOINTS, BEST_LOSS, device


def accuracy(output, target):
    accuracy_value = 0
    e = 20
    output = output.reshape(-1, NUM_KEYPOINTS // 2, 2)
    target = target.reshape(-1, NUM_KEYPOINTS // 2, 2)

    for i in range(target.shape[0]):
        acc = 0
        for j in range(NUM_KEYPOINTS // 2):
            if (abs(output[i][j][0].item() - target[i][j][0].item()) < e) and (
                abs(output[i][j][1].item() - target[i][j][1].item()) < e
            ):
                acc += 1
        if acc == NUM_KEYPOINTS // 2:
            accuracy_value += 1
    return accuracy_value


# training function
def train(model, dataloader, data, criterion, optimizer, criterion_point_class):
    print("Training")
    model.train()
    train_running_accuracy = 0.0
    train_running_loss = 0.0
    counter = 0
    # calculate the number of batches
    num_batches = int(len(data) / dataloader.batch_size)
    for i, data in tqdm(enumerate(dataloader), total=num_batches):
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
        """
        # acuracy
        for i in range(heatmaps.shape[0]):
            pred_indices_linear, pred_Sxy, head = decode_card(
                heatmaps[i],
                offsets[i])
            pred = image_to_square(pred_indices_linear, pred_Sxy)
            train_running_accuracy += accuracy(pred, keypoints[i])"""

        loss.backward()
        optimizer.step()

    train_accuracy = train_running_accuracy
    train_loss = train_running_loss / counter
    return train_accuracy, train_loss


# validatioon function
def validate(model, dataloader, data, criterion, criterion_point_class, best_loss=BEST_LOSS):
    print("Validating")
    model.eval()
    val_running_accuracy = 0.0
    val_running_loss = 0.0
    counter = 0
    # calculate the number of batches
    num_batches = int(len(data) / dataloader.batch_size)
    for i, data in tqdm(enumerate(dataloader), total=num_batches):
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

    # deep copy the model
    if val_running_loss < best_loss:
        best_loss -= best_loss - val_running_loss
        best_model_wts = copy.deepcopy(model.state_dict())

    val_accuracy = val_running_accuracy
    if counter == 0:
        val_loss = 0
    else:
        val_loss = val_running_loss / counter
    return val_accuracy, val_loss, best_model_wts
