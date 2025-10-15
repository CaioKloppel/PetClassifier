import sys
import torch
from PIL import Image
from torchvision import transforms
from modelArchitecture.modelArchitecture import CNN

def load_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        return image
    except Exception as e:
        print(f"Image of adress '{image_path}' not found.")

def image_to_tensor(image):
    transform_pipeline = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
    return transform_pipeline(image)

def returnModel(adress, device):
    model = CNN()
    model.load_state_dict(torch.load(adress))
    model = model.to(device)
    return model

def main():
        if len(sys.argv) != 2:
            print("Invalid number of arguments!\nCorrect usage: ./predict <path/to/photo>")
            return 1
        
        image = load_image(sys.argv[1])
        if image: 
            print('='*50)
            print(f'ANALYZING IMAGE: {sys.argv[1]}...'.center(50))
            print('='*50)
            tensor = image_to_tensor(image)
        else: return 1 

        tensor = tensor.view(-1, 3, tensor.shape[1], tensor.shape[2])
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = returnModel('modelPet6.pth', device)

        model.eval()
        
        tensor = tensor.to(device)
        output = model(tensor)
        predicted = (output > 0.5).float()

        print("IT'S A CAT".center(50) if predicted[0].item() == 0 else "IT'S A DOG".center(50))
        print('='*50 + '\n\n\n')

        input("Press Enter to finish!")

        del model
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()

