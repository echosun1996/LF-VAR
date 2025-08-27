import os
import argparse

def parse_metrics(file_path):
    
    with open(file_path, 'r') as file:
        metrics = {}
        for line in file:
            key, value = line.strip().split(': ')
            metrics[key] = float(value)
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Process and print metrics.")
    parser.add_argument("--metrics_dir", required=True, help="Directory containing metrics files.")
    parser.add_argument("--evaluate", required=True, help="Prefix to filter files for evaluation.")
    parser.add_argument("--compare", required=True, help="Compare with train/test/val dataset.")
    args = parser.parse_args()

    files = [
        f for f in os.listdir(args.metrics_dir)
        if f.startswith(args.evaluate+"_") and f.endswith(args.compare+".txt")
    ]
    print("Start to print the metrics for [",args.evaluate,"]:")
    if not files:
        print("No matching files found.")
        return
    means = []
    stds = []
    fids = []
    for file_name in files:
        file_path = os.path.join(args.metrics_dir, file_name)
        metrics = parse_metrics(file_path)
        fid = metrics.get("FID", 0.0)
        fids.append(fid)
        print(file_name.replace(args.evaluate+"_", "").replace(".txt","").replace("_test","")+":",end="")
        print(f"{fid:.2f}")

    print("=============")


if __name__ == "__main__":
    main()