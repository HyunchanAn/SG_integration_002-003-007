import torch
import argparse
import os

def export_depth_anything_onnx(checkpoint_path, encoder, output_path):
    try:
        from src.topo.depth_anything_v2.dpt import DepthAnythingV2
    except ImportError:
        print("Error: Could not import DepthAnythingV2. Make sure you are in the project root.")
        return

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        return

    print(f"Loading Depth-Anything-V2 ({encoder}) from {checkpoint_path}...")
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()
    
    # Depth-Anything-V2 standard input shape is (1, 3, 518, 518)
    dummy_input = torch.randn(1, 3, 518, 518)
    
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path,
        export_params=True, 
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                      'output': {0: 'batch_size', 1: 'height', 2: 'width'}}
    )
    print("Export successful. You can now use ONNXRuntime for edge inference.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch models to ONNX for edge deployment.")
    parser.add_argument('--ckpt', type=str, default='models/depth_anything_v2/depth_anything_v2_vits.pth', help='Path to PyTorch checkpoint')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl'], help='Model encoder type')
    parser.add_argument('--out', type=str, default='models/depth_anything_v2.onnx', help='Output ONNX file path')
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    export_depth_anything_onnx(args.ckpt, args.encoder, args.out)
