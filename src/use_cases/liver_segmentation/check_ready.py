"""
FedMorph Client Readiness Check
=================================

Run this on each client PC to verify everything is ready
for federated learning BEFORE starting actual training.

Checks:
  1. Python dependencies (torch, monai, flwr, etc.)
  2. GPU availability & VRAM
  3. Local data validity
  4. Server connectivity (gRPC port)

Usage:
  uv run python src/use_cases/liver_segmentation/check_ready.py \
      --data-dir D:\data\liver_ct \
      --server-address 192.168.1.100:443

  # data check only (no server)
  uv run python src/use_cases/liver_segmentation/check_ready.py \
      --data-dir D:\data\liver_ct
"""

import argparse
import os
import socket
import sys
import time

REQUIRED_PACKAGES = {
    "torch": "torch",
    "numpy": "numpy",
    "monai": "monai",
    "flwr": "flwr",
    "yaml": "pyyaml",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
}


def check_dependencies():
    print("[1/4] Dependencies")
    all_ok = True
    for import_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            __import__(import_name)
            print(f"  OK   {import_name}")
        except ImportError:
            print(f"  FAIL {import_name}  →  uv pip install {pip_name}")
            all_ok = False
    return all_ok


def check_gpu():
    print("\n[2/4] GPU")
    try:
        import torch
        if not torch.cuda.is_available():
            print("  WARN  CUDA not available — will use CPU (very slow)")
            return True

        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1024**3
        print(f"  OK   {props.name}")
        print(f"  OK   VRAM: {vram_gb:.1f} GB")

        if vram_gb < 4:
            print(f"  WARN  VRAM {vram_gb:.1f} GB may be insufficient (recommend >= 6 GB)")
        else:
            print(f"  OK   VRAM sufficient")

        if props.major >= 8:
            print(f"  OK   Compute capability {props.major}.{props.minor} (TF32 supported)")
        else:
            print(f"  OK   Compute capability {props.major}.{props.minor}")

        free_mem = torch.cuda.mem_get_info(0)
        print(f"  OK   Free VRAM: {free_mem[0] / 1024**3:.1f} / {free_mem[1] / 1024**3:.1f} GB")
        return True
    except Exception as e:
        print(f"  FAIL  GPU check error: {e}")
        return False


def check_data(data_dir):
    print("\n[3/4] Local Data")
    if not data_dir:
        print("  SKIP  --data-dir not specified")
        return True

    if not os.path.isdir(data_dir):
        print(f"  FAIL  Directory not found: {data_dir}")
        return False

    print(f"  OK   Directory exists: {data_dir}")

    patient_count = 0
    ok_count = 0
    issues = []

    for name in sorted(os.listdir(data_dir)):
        patient_dir = os.path.join(data_dir, name)
        if not os.path.isdir(patient_dir):
            continue

        img_path = os.path.join(patient_dir, "image.npy")
        mask_path = os.path.join(patient_dir, "mask.npy")

        if not os.path.isfile(img_path) or not os.path.isfile(mask_path):
            continue

        patient_count += 1
        try:
            import numpy as np
            img = np.load(img_path)
            mask = np.load(mask_path)
            if img.ndim == 3 and mask.ndim == 4 and mask.shape[0] >= 10:
                ok_count += 1
            else:
                issues.append(f"{name}: image {img.shape}, mask {mask.shape}")
        except Exception as e:
            issues.append(f"{name}: {e}")

    if patient_count == 0:
        print(f"  FAIL  No patients found (need subfolders with image.npy + mask.npy)")
        return False

    print(f"  OK   {patient_count} patients found, {ok_count} valid")

    if issues:
        for issue in issues[:5]:
            print(f"  WARN  {issue}")
        if len(issues) > 5:
            print(f"  WARN  ... and {len(issues) - 5} more")

    if ok_count < 2:
        print(f"  FAIL  Need at least 2 valid patients for train/val split")
        return False

    return True


def check_server(server_address):
    print("\n[4/4] Server Connectivity")
    if not server_address:
        print("  SKIP  --server-address not specified")
        return True

    if ":" not in server_address:
        print(f"  FAIL  Invalid format: {server_address} (expected HOST:PORT)")
        return False

    host, port_str = server_address.rsplit(":", 1)
    try:
        port = int(port_str)
    except ValueError:
        print(f"  FAIL  Invalid port: {port_str}")
        return False

    # TCP connectivity test
    print(f"  ...  Testing TCP connection to {host}:{port}")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        start = time.time()
        result = sock.connect_ex((host, port))
        elapsed = time.time() - start
        sock.close()

        if result == 0:
            print(f"  OK   TCP port {port} is open ({elapsed*1000:.0f} ms)")
        else:
            print(f"  FAIL  TCP port {port} is closed or unreachable")
            print(f"        Server must be running first: run_liver_server.bat")
            print(f"        Server firewall must allow port {port}")
            return False
    except socket.timeout:
        print(f"  FAIL  Connection timed out (5s)")
        print(f"        Check: server IP correct? firewall open?")
        return False
    except socket.gaierror:
        print(f"  FAIL  Cannot resolve hostname: {host}")
        return False
    except Exception as e:
        print(f"  FAIL  {e}")
        return False

    # gRPC test
    try:
        import grpc
        print(f"  ...  Testing gRPC connection to {server_address}")
        channel = grpc.insecure_channel(server_address)
        try:
            grpc.channel_ready_future(channel).result(timeout=5)
            print(f"  OK   gRPC server is responding")
        except grpc.FutureTimeoutError:
            print(f"  WARN  gRPC handshake timed out (server may not be started yet)")
            print(f"        TCP port is open, so network path is OK")
            print(f"        Start the server first, then this check will pass")
        finally:
            channel.close()
    except ImportError:
        print(f"  SKIP  grpc not installed, TCP check passed")
    except Exception as e:
        print(f"  WARN  gRPC test error: {e}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="FedMorph Client Readiness Check")
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Local CT data directory")
    parser.add_argument(
        "--server-address", type=str, default=None,
        help="Server address (e.g. 192.168.1.100:443)")
    args = parser.parse_args()

    print("=" * 60)
    print("  FedMorph — Client Readiness Check")
    print("=" * 60)
    print()

    results = {}
    results["deps"] = check_dependencies()
    results["gpu"] = check_gpu()
    results["data"] = check_data(args.data_dir)
    results["server"] = check_server(args.server_address)

    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")

    labels = {
        "deps": "Dependencies",
        "gpu": "GPU",
        "data": "Local Data",
        "server": "Server",
    }
    all_pass = True
    for key, label in labels.items():
        status = "PASS" if results[key] else "FAIL"
        icon = "OK" if results[key] else "!!"
        print(f"  [{icon}] {label:<20} {status}")
        if not results[key]:
            all_pass = False

    print()
    if all_pass:
        cmd_parts = ["uv run python src/use_cases/liver_segmentation/main_client.py"]
        if args.server_address:
            cmd_parts.append(f"--server-address {args.server_address}")
        if args.data_dir:
            cmd_parts.append(f"--data-dir {args.data_dir}")
        print("  All checks passed! Start with:")
        print(f"    {' '.join(cmd_parts)}")
    else:
        print("  Some checks failed. Fix the issues above before starting.")

    print()


if __name__ == "__main__":
    main()
