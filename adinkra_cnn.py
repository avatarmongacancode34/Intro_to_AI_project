class AdinkraCNN(nn.Module):
  def __init__(self, num_classes=10):
    super(AdinkraCNN, self).__init__()

    # creating the different blocks of the CNN

    # edges block
    self.conv1 = nn.Conv2d(in_channels = 3, out_channels=16,kernel_size=3,padding=1)
    self.pool1= nn.MaxPool2d(kernel_size=2,stride=2)

    #corners and curves
    self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1)
    self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)

    # complex shapes of the symbol
    self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
    self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2)

    # fully connected layers
    self.fc1 = nn.Linear(in_features=64*28*28,out_features=128)

    ## apply droup out to 50% of the neurons
    self.dropout = nn.Dropout(p = 0.5)

    #final output
    self.fc2 = nn.Linear(in_features=128,out_features=num_classes)
  def forward(self,x):
    x = self.pool1(F.relu(self.conv1(x)))
    x = self.pool2(F.relu(self.conv2(x)))
    x = self.pool3(F.relu(self.conv3(x)))

    # flattern
    x = torch.flatten(x,1)

    # decision layer
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)

    return x
if __name__ == "__main__":
  model = AdinkraCNN(num_classes=10)
  print(model)

  

