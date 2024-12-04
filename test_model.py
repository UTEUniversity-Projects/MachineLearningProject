import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models.my_model import *  # Import model class

# Load the checkpoint
def load_model(checkpoint_path, num_classes=3):
    print(f"Loading model from {checkpoint_path}")
    model = Net01(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    # Remove 'module.' prefix if it exists in state_dict
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")  # Remove `module.` prefix
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model.eval()  # Set model to evaluation mode
    return model

# Evaluate model on test dataset
def evaluate_model(model, test_loader, classes, result_file_path):
    class_correct = [0 for _ in range(len(classes))]
    class_total = [0 for _ in range(len(classes))]
    
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # Write results to file
    with open(result_file_path, 'w') as result_file:
        for i in range(len(classes)):
            if class_total[i] > 0:
                accuracy = 100 * class_correct[i] / class_total[i]
                result_file.write(f'Accuracy of {classes[i]}: {accuracy:.2f}%\n')
                print(f'Accuracy of {classes[i]} : {accuracy:.2f} %')
            else:
                result_file.write(f'Accuracy of {classes[i]}: N/A (no samples)\n')
                print(f'Accuracy of {classes[i]} : N/A (no samples)')

def main():
    checkpoint_path = "./checkpoints/MyNet_Dog_Cat_Squirrel_Net01_03_/model_best.pth.tar"
    data_path = "./dataset/ImageNet3Classes/val"
    batch_size = 16
    num_classes = 3
    classes = ['Dog', 'Cat', 'Squirrel']

    # Define result file path
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)  # Create directory if it doesn't exist
    result_file_path = os.path.join(results_dir, "MyNet_Dog_Cat_Squirrel_Net01_03.txt")

    # Define data transformations for test dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Load test dataset
    test_dataset = datasets.ImageFolder(data_path, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model = load_model(checkpoint_path, num_classes=num_classes)

    # Evaluate model and write results to file
    evaluate_model(model, test_loader, classes, result_file_path)

if __name__ == "__main__":
    main()