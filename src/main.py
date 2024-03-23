from src.models.CLIP import CLIPWithHead
import pdb

if __name__ == '__main__':
    CWH = CLIPWithHead()
    output = CWH.example()
    print(f'Output keys: {output.keys()}')
    for name, param in CWH.named_parameters():
        if param.requires_grad:
            print(f"Parameter name which requires grad: {name}")
    