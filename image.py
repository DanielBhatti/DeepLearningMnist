from PIL import Image
import torch

@static
class Reader:
    @staticmethod
    def image_to_tensor(file_path: str):
        image = Image.open(file_path)
        pixels = image.load()
        print(image)

if __name__ == "__main__":
    Reader.image_to_tensor("") 