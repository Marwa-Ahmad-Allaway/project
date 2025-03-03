import argparse
from torchvision import transforms, models
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import json

def process_img(image):
  input_image = Image.open(image)
  image_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
  image_processed = image_transforms(input_image)
  return image_processed


def predect_img(img_path, model, topk,category_names):
  images_valid = process_img(img_path)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  model.eval()
  with torch.no_grad():
    log_ps = model(images_valid)
    ps = torch.exp(log_ps)
    top_p, top_class = ps.topk(topk)
    with open(category_names) as f:
      data = f.read()
      js = json.loads(data)
      print(top_p)
      print(top_class)
      print(js[top_class[i]]for i in range(topk))


def get_input_arg():
  parser = argparse.ArgumentParser()
  parser.add_argument('--top_k',type=int,default=1)
  parser.add_argument('--category_names',type=str,default="cat_to_name.json")
def main():
  image_path = str(input())
  checkpoint = str(input())
  in_arg = get_input_arg()
  predect_img(image_path,checkpoint,in_arg.top_k,in_arg.category_names)
if __name__ == "__main__":
  main()
