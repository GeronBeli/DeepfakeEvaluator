import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import cv2
import face_recognition
import dlib
import numpy as np



class DeepfakeDataset(Dataset):
    """Dataset for loading deepfake detection images."""
    
    def __init__(self, root_dir, transform, face_detection):
        """
        Args:
            root_dir: Directory with REAL/ and FAKE/ subdirectories
            transform: Optional transform to be applied to images
        """
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Get all image paths and labels
        self.samples = []
        self.face_detection = face_detection

        
        # Real images - label 0
        real_dir = os.path.join(root_dir, "REAL")
        for img_name in os.listdir(real_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(real_dir, img_name)
                self.samples.append((img_path, 0))
        
        # Fake images - label 1
        fake_dir = os.path.join(root_dir, "FAKE")
        for img_name in os.listdir(fake_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(fake_dir, img_name)
                self.samples.append((img_path, 1))
    
    def __len__(self):
        return len(self.samples)
    
    def extract_face(self, image):
        """Detects and extracts the face from an image."""
        mod = "cnn" if dlib.DLIB_USE_CUDA else "hog"

        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        face_locations = face_recognition.face_locations(image_bgr, model=mod)

        if not face_locations:
            return None  # No face found

        top, right, bottom, left = face_locations[0]
        face_image = image_bgr[top:bottom, left:right]  
        face_image = cv2.resize(face_image, (224, 224), interpolation=cv2.INTER_AREA)  
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        return face_image


    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'label': label,
            'path': img_path
        }

def get_dataloader(root_dir, batch_size=1, shuffle=False, transform=None, face_detection= True):
    """Create and return a DataLoader for evaluation."""
    dataset = DeepfakeDataset(root_dir, transform, face_detection=face_detection)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
