from DMCNN_new import *
from cv2 import imread, resize
from os import listdir, mkdir
from os.path import isfile, join, isdir, exists


x = []
y = []


mypath = "filtered_dataset/training"
origin_mypath = "origin_filtered_dataset/training"
evalpath = "filtered_dataset/validation"
origin_evalpath = "origin_filtered_dataset/validation"


def add_channels(img):
    img[:, :, 1] = 0
    img[:, :, 2] = 0
    return img

onlyfiles = [f for f in listdir(join(mypath)) if isfile(join(mypath, f))]
for f in onlyfiles:
    im = imread(join(mypath, f))
    im = add_channels(im)
    x.append(im)
    y.append(imread(join(origin_mypath, f)))



x = np.array(x)
y = np.array(y)
randomize = np.arange(len(x))
np.random.shuffle(randomize)
x = x[randomize]
y = y[randomize]

x_eval = []
y_eval = []

onlyfiles = [f for f in listdir(join(evalpath)) if isfile(join(evalpath, f))]
for f in onlyfiles:
    im = imread(join(evalpath, f))
    im = add_channels(im)
    x_eval.append(im)
    y_eval.append(imread(join(origin_evalpath, f)))

x_eval = np.array(x_eval)
y_eval = np.array(y_eval)
randomize = np.arange(len(x_eval))
np.random.shuffle(randomize)
x_eval = x_eval[randomize]
y_eval = y_eval[randomize]

model = DMCNN()
summary(model, input_size=(3, 33, 33))
device = torch.device('cuda:0')
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(
    params = [
        {"params": model.feature_layer.parameters(), "lr": 1},
        {"params": model.mapping_layer.parameters(), "lr": 1},
        {"params": model.reconstruction_layer.parameters(), "lr": 0.1}
    ]
)

n_epochs = 500
total_step = len(x)
loss_list = []
counter = 0
for epoch in range(n_epochs):
    epoch_loss = []
    for i in range(len(x)):
        cfa = cfa.float().to(device)
        target = target.float().to(device)

        outputs = model(cfa)
        loss = criterion(outputs, target)
        loss_list.append(loss.item())
        epoch_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        counter += 1
        if counter % 10 == 0:
            # print(outputs)
            print(f'Epoch [{epoch}/{n_epochs}], Step [{counter}/{total_step}], Loss: {loss.item()}')
    epoch_stats = np.array(epoch_loss)
    print(f'\nFinished Epoch {epoch}, Loss --- mean: {epoch_stats.mean()}, std {epoch_stats.std()}\n')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 8))
    ax1.imshow(np.array(outputs[-1].tolist()).reshape((21, 21, 3)))
    ax2.imshow(np.array(cfa[-1].tolist()).reshape((33, 33, 3)))
    ax3.imshow(np.array(target[-1].tolist()).reshape((21, 21, 3)))
    plt.show()