import os
import torch
from torchvision import transforms, datasets, models
import torch.optim as optim
import torch.nn as nn

# Define the number of epochs
epochs = 3
steps = 0
running_loss = 0
print_every = 10
train_losses, test_losses = [], []

# Define transforms for the training, validation, and testing sets
train_transform = transforms.Compose([
    transforms.RandomRotation(50),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_transform = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define data directories
data_dir = 'flowers'
train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'valid')
test_dir = os.path.join(data_dir, 'test')

train_data = datasets.ImageFolder(train_dir, transform=train_transform)
test_data = datasets.ImageFolder(test_dir, transform=test_transform)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transform)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model
model = models.resnet50(pretrained=True)
model.to(device)

# Define the optimizer
optimizer = optim.Adam(model.fc.parameters(), lr=0.0025)

# Define the criterion
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            
            model.eval()
            
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    valid_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_ps, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            train_losses.append(running_loss/print_every)
            test_losses.append(valid_loss/len(testloader))
                          
            print(f"Epoch {epoch+1}/{epochs}.. "
                f"Train loss: {running_loss/print_every:.3f}.. "
                f"Test loss: {valid_loss/len(testloader):.3f}.. "
                f"Test accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()

# Save the trained model in the current directory
current_directory = os.getcwd()
model_filename = 'unique_special.pth'
model_path = os.path.join(current_directory, model_filename)

# Save model's state_dict and other necessary information
torch.save({
    'extraordinary_model': model.state_dict(),
    'extraordinary_classifier': model.fc,
    'class_to_idx': train_data.class_to_idx
}, model_path)