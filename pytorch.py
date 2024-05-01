import torch
import numpy as np

np.set_printoptions(precision=3)
a = [1,2,3]
b = np.array([4,5,6], dtype=np.int32)
t_a = torch.tensor(a)
t_b = torch.from_numpy(b)
print(t_a)
print(t_b)

t_ones = torch.ones(2,3)
print(t_ones.shape)
print(t_ones)

rand_tensor = torch.rand(2,3)
print(rand_tensor)

t_a_new = t_a.to(torch.int64)
print(t_a_new.dtype)

t = torch.rand(3,5)
t_tr = torch.transpose(t, 0, 1)
print(t.shape, '-->', t_tr.shape)

t = torch.zeros(30)
t_reshape = t.reshape(5, 6)
print(t_reshape.shape)

t = torch.zeros(1,2,1,4,1)
t_sqz = torch.squeeze(t, 2)
print(t.shape, '-->', t_sqz.shape)


## applying mathematical operations to tensors

torch.manual_seed(1)
t1 = 2 * torch.rand(5, 2) - 1  # uniform distribution [-1, 1)
t2 = torch.normal(mean=0, std=1, size=(5,2))  # standard normal distribution

# element-wise product
t3 = torch.multiply(t1, t2)
print(t3)

t4 = torch.mean(t1, axis=0)
print(t4)

t5 = torch.matmul(t1, torch.transpose(t2, 0, 1))
print(t5)

t6 = torch.matmul(torch.transpose(t1, 0, 1), t2)
print(t6)

norm_t1 = torch.linalg.norm(t1, ord=2, dim=1)
print(norm_t1)
# compare with:
print(np.sqrt(np.sum(np.square(t1.numpy()), axis=1)))

## split, stack, concatenate tensors

torch.manual_seed(1)
t = torch.rand(6)
print(t)
t_splits = torch.chunk(t, 3)  # define the number of splits
print([item.numpy() for item in t_splits])

torch.manual_seed(1)
t = torch.rand(5)
print(t)
t_splits = torch.split(t, split_size_or_sections = [3,2]) # specify size of output tensors
print([item.numpy() for item in t_splits])


# concatenate a 1D tensor size 3 with 1D tensor size 2 = 1D tensor size 5
A = torch.ones(3)
B = torch.zeros(2)
C = torch.cat([A,B], axis=0)
print(C)


# create 1D tensors A & B size 3, to form a 2D tensor
A = torch.ones(3)
B = torch.zeros(3)
S = torch.stack([A, B], axis=1)
print(S)
print(S.shape)


# data loader
from torch.utils.data import DataLoader
t = torch.arange(6, dtype=torch.float32)
data_loader = DataLoader(t)
for item in data_loader:
    print(item)

data_loader = DataLoader(t, batch_size=3, drop_last=False)
for i, batch in enumerate(data_loader, 1):
    print(f'batch {i}:', batch)


# combining two tensors into a joint dataset
torch.manual_seed(1)
t_x = torch.rand([4, 3], dtype=torch.float32)
t_y = torch.arange(4)

from torch.utils.data import Dataset
class JointDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

joint_dataset = JointDataset(t_x, t_y)

for example in joint_dataset:
    print(' x:', example[0], ' y:', example[1])

# alternatively use the torch TensorDataset class:
from torch.utils.data import TensorDataset
joint_dataset2 = TensorDataset(t_x, t_y)

# shuffle, batch, and repeat

torch.manual_seed(1)
data_loader = DataLoader(dataset=joint_dataset, batch_size=2, shuffle=True)

for i, batch in enumerate(data_loader, 1):
    print(f'batch {i}:', 'x:', batch[0], '\n   y:', batch[1])

for epoch in range(2):
    print(f'epoch {epoch + 1}')
    for i, batch in enumerate(data_loader, 1):
        print(f'batch {i}:', 'x:', batch[0], '\n     y:', batch[1])


# create dataset from files on local storage disk
import pathlib
imgdir_path = pathlib.Path('cat_dog_images')
file_list = sorted([str(path) for path in imgdir_path.glob('*.jpg')])
print(file_list)

import matplotlib.pyplot as plt
import os
from PIL import Image
fig = plt.figure(figsize=(10,5))
for i, file in enumerate(file_list):
    img = Image.open(file)
    print('Image shape:', np.array(img).shape)
    ax = fig.add_subplot(2, 3, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(img)
    ax.set_title(os.path.basename(file), size=15)
plt.tight_layout()
plt.show()

labels = [1 if 'dog' in os.path.basename(file) else 0 for file in file_list]
print(labels)

# now we have two lists: filenames and labels.  create a joint dataset.
class ImageDataset(Dataset):
    def __init__(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.file_list[index])
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.labels)


# transform images and covert the loaded pixels into tensors
import torchvision.transforms as transforms
img_height, img_width = 80, 120
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((img_height, img_width)),
])

image_dataset = ImageDataset(file_list, labels, transform)

fig = plt.figure(figsize=(10, 6))
for i, example in enumerate(image_dataset):
    ax = fig.add_subplot(2, 3, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(example[0].numpy().transpose((1, 2, 0)))
    ax.set_title(f'{example[1]}', size=15)
plt.tight_layout()
plt.show()


# download CelebA dataset
import torchvision
image_path = './'

### I think the annotations need to be added - there are no txt files in the directory
#celeba_dataset = torchvision.datasets.CelebA(
#    image_path, split='train', target_type='attr', download=False
#)
#assert isinstance(celeba_dataset, torch.utils.data.Dataset)

#example = next(iter(celeba_dataset))
#print(example)
#
# # comes as (PIL.Image, attributes).  reformat to (features tensor, label)
from itertools import islice
# fig = plt.figure(figsize=(12, 18))
# for i, (image, attributes) in islice(enumerate(celeba_dataset), 18):
#     ax, fig.add_subplot(3, 6, i + 1)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.imshow(image)
#     ax.set_title(f'{attributes[31]}', size=15)
#     plt.show()


# MNIST dataset
mnist_dataset = torchvision.datasets.MNIST(image_path, 'train', download=True)
assert isinstance(mnist_dataset, torch.utils.data.Dataset)
example = next(iter(mnist_dataset))
print(example)

fig = plt.figure(figsize=(15, 6))
for i, (image, label) in islice(enumerate(mnist_dataset), 10):
    ax = fig.add_subplot(2, 5, i + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(image, cmap='gray_r')
    ax.set_title(f'{label}', size=15)
plt.show()


## building a linear regression model

X_train = np.arange(10, dtype='float32').reshape((10, 1))
y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0,
                    6.3, 6.6, 7.4, 8.0,
                    9.0], dtype='float32')
plt.plot(X_train, y_train, 'o', markersize=10)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# standardize the features - mean centering and diving by s.d., and creating
# a PyTorch Dataset for training and a corresponding DataLoader

from torch.utils.data import TensorDataset
X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)
X_train_norm = torch.from_numpy(X_train_norm)
y_train = torch.from_numpy(y_train).float()
train_ds = TensorDataset(X_train_norm, y_train)
batch_size = 1
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# define a linear regression model from scratch (will use predefined layers later)
torch.manual_seed(1)
weight = torch.randn(1)
weight.requires_grad_()
bias = torch.zeros(1, requires_grad=True)


def model(xb):
    return torch.tensor(xb) @ weight + bias


def loss_fn(input, target):
   return (input-target).pow(2).mean()


# train using SGD - later using torch.optim package
learning_rate = 0.001
num_epochs = 200
log_epochs = 10
for epoch in range(num_epochs):
    for x_batch, y_batch in train_dl:
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch.long())
        loss.backward()
    with torch.no_grad():
        weight -= weight.grad * learning_rate
        bias -= bias.grad * learning_rate
        weight.grad.zero_()
        bias.grad.zero_()
    if epoch % log_epochs == 0:
        print(f'Epoch {epoch} Loss {loss.item():.4f}')

# look at trained model and plot
print('Final Parameters:', weight.item(), bias.item())
X_test = np.linspace(0, 9, num=100, dtype='float32').reshape(-1, 1)
X_test_norm = (X_test - np.mean(X_train)) / np.std(X_train) # apply same standardization
y_pred = model(X_test_norm).detach().numpy()
fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(1,2,1)
plt.plot(X_train_norm, y_train, 'o', markersize=10)
plt.plot(X_test_norm, y_pred, '--', lw=3)
plt.legend(['Training examples', 'Linear reg.'], fontsize=15)
ax.set_xlabel('x', size=15)
ax.set_ylabel('y', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.show()


# now using torch

import torch.nn as nn
loss_fn = nn.MSELoss(reduction='mean')
input_size = 1
output_size = 1
model = nn.Linear(input_size, output_size)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for x_batch, y_batch in train_dl:
        # 1. Generate predictions
        pred = model(x_batch)[:, 0]
        # 2. Calculate loss
        loss = loss_fn(pred, y_batch)
        # 3. Compute gradients
        loss.backward()
        # 4. Update parameters using gradients
        optimizer.step()
        # 5. Reset the gradients to zero
        optimizer.zero_grad()
    if epoch % log_epochs == 0:
        print(f'Epoch {epoch} Loss {loss.item():.4f}')

print('Final Parameters:', model.weight.item(), model.bias.item())


# using torch predefined layers to classify flowers
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()
X = iris['data']
y = iris['target']
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=1./3, random_state=1)

X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)
X_train_norm = torch.from_numpy(X_train_norm).float()
y_train = torch.from_numpy(y_train)
train_ds = TensorDataset(X_train_norm, y_train)
torch.manual_seed(1)
batch_size = 2
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = self.layer1(x)
        x = nn.Sigmoid()(x)
        x = self.layer2(x)
        return x

input_size = X_train_norm.shape[1]
hidden_size = 16
output_size = 3
model = Model(input_size, hidden_size, output_size)

learning_rate = 0.001
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 100
loss_hist = [0] * num_epochs
accuracy_hist = [0] * num_epochs
for epoch in range(num_epochs):
    for x_batch, y_batch in train_dl:
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch.type(torch.long))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_hist[epoch] += loss.item() * y_batch.size(0)
        is_correct = (torch.argmax(pred, dim=1)==y_batch).float()
        accuracy_hist[epoch] += is_correct.sum()
    loss_hist[epoch] /= len(train_dl.dataset)
    accuracy_hist[epoch] /= len(train_dl.dataset)

# visualize the learning curves
fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(1,2,1)
ax.plot(loss_hist, lw=3)
ax.set_title('Training loss', size=15)
ax.set_xlabel('Epoch', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)
ax = fig.add_subplot(1,2,2)
ax.plot(accuracy_hist, lw=3)
ax.set_title('Training accuracy', size=15)
ax.set_xlabel('Epoch', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.show()

# evaluate on test dataset
X_test_norm = (X_test - np.mean(X_train)) / np.std(X_train)
X_test_norm = torch.from_numpy(X_test_norm).float()
y_test = torch.from_numpy(y_test)
pred_test = model(X_test_norm)
correct = (torch.argmax(pred_test, dim=1)==y_test).float()
accuracy = correct.mean()
print(f'Test Acc.: {accuracy:.4f}')


# save and reload model
path = 'iris_classifier.pt'
torch.save(model, path)
model_new = torch.load(path)
print(model_new.eval())

# verify the reloaded model gives the same results
pred_test = model_new(X_test_norm)
correct = (torch.argmax(pred_test,dim=1)==y_test).float()
accuracy = correct.mean()
print(f'Test Acc.: {accuracy:.4f}')

# to save/reload just parameters (not model):
path = 'iris_classifier_state.pt'
torch.save(model.state_dict(),path)

model_new = Model(input_size, hidden_size, output_size)
model_new.load_state_dict(torch.load(path))


