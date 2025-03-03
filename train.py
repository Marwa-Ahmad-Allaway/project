import argparse
from torchvision import transforms, models
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
def transdata(data):
  data_dir = 'flower_data'
  train_dir = data_dir + '\\train'
  valid_dir = data_dir + '\\valid'
  test_dir = data_dir + '\\test'
  data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                        ])
  test_transforms = transforms.Compose([transforms.Resize(224),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
  image_datasets = datasets.ImageFolder(data_dir, transform=data_transforms)
  train_data= datasets.ImageFolder(train_dir,transform=data_transforms)
  valid_data = datasets.ImageFolder(valid_dir,transform=test_transforms)
  dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=512, shuffle=True)
  train_loader = torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True)
  valid_loader = torch.utils.data.DataLoader(valid_data,batch_size=64)
  return train_loader,valid_loader


def train_model(data, premodel , saved_direc ,learning_rate  ,hidden , epochs ):
  train_loader,valid_loader = transdata(data)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  if premodel == "vgg11":
    model = models.vgg11(pretrained=True)
  elif premodel == "vgg13":
    model = models.vgg13(pretrained=True)
  elif premodel == "vgg16":
    model = models.vgg16(pretrained=True)
  else:
    raise ValueError("Unsupported model architecture")
  for param in model.parameters():
    param.requires_grad = False
  classifier = nn.Sequential(nn.Linear(25088, hidden),
                            nn.ReLU(),
                            nn.Linear(hidden, 102),
                            nn.Dropout(0.2),
                            nn.LogSoftmax(dim=1))
  model.classifier = classifier
  criterion = nn.NLLLoss()
  optimizer = optim.Adam(model.classifier.parameters(), lr =learning_rate)
  model.to(device)
  print_every = 64
  for e in range(epochs):
      steps = 0
      running_loss = 0
      for images, labels in train_loader:
          steps +=1
          images, labels = images.to(device), labels.to(device)
          optimizer.zero_grad()

          log_ps = model(images)
          loss = criterion(log_ps, labels)
          loss.backward()
          optimizer.step()
          running_loss += loss.item()

          if steps % print_every == 0:
              test_loss = 0
              accuracy = 0
              model.eval()
              # Turn off gradients for validation, saves memory and computations
              with torch.no_grad():
                  for images_valid, labels_valid in valid_loader:
                      images_valid, labels_valid = images_valid.to(device), labels_valid.to(device)
                      log_ps = model(images_valid)
                      batch_loss = criterion(log_ps, labels_valid)
                      test_loss += batch_loss.item()
                      ps = torch.exp(log_ps)
                      top_p, top_class = ps.topk(1)
                      equals =( top_class == (labels_valid.view(*top_class.shape)))
                      accuracy += equals.type(torch.FloatTensor).sum().item()

              print(f"Epoch {e+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Test loss: {test_loss/len(valid_loader):.3f}.. "
                    f"Test accuracy: {accuracy/len(valid_loader):.3f}")
              running_loss = 0
              model.train()
  return model

def get_input_arg():
  parser = argparse.Argumentparser()
  parser.add_argument('--save_dir',type=str,default="home/workspace/")
  parser.add_argument('--arch',type=str,default="vgg16")
  parser.add_argument('--learning_rate',type=int,default=0.01)
  parser.add_argument('--hidden_units',type=int,default=512)
  parser.add_argument( '--epochs',type=int,default=20)
def main():
  data_dir = str(input())
  in_arg = get_input_arg()
  model = train_model(data_dir,in_arg.arch,in_arg.save_dir,in_arg.learning_rate,in_arg.hidden_units,in_arg.epochs)
  # TODO: Save the checkpoint
  checkpoint = {'model': in_arg.arch,
                'input_size': 25088,
                'output_size': 102,
                'state_dict': model.state_dict()}
  torch.save(checkpoint, in_arg.save_dir + 'train_checkpoint.pth')
if __name__ = "__main__":
  main()
