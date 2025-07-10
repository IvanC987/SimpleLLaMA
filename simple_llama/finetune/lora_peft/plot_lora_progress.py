import matplotlib.pyplot as plt

fpath1 = "r1.txt"
fpath2 = "r16.txt"

def parse_avg_loss(lines, group_size=5):
    header = lines[0].split(",")
    idx_loss = header.index("Training Loss")

    losses = []
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.strip().split(",")
        losses.append(float(parts[idx_loss]))

    # 5:1 averaging
    grouped = [
        sum(losses[i:i + group_size]) / len(losses[i:i + group_size])
        for i in range(0, len(losses), group_size)
        if len(losses[i:i + group_size]) == group_size
    ]
    return grouped

with open(fpath1, "r") as f:
    lines1 = f.read().splitlines()

with open(fpath2, "r") as f:
    lines2 = f.read().splitlines()

y1 = parse_avg_loss(lines1)
y2 = parse_avg_loss(lines2)

plt.figure(figsize=(8, 5))
plt.plot(y1, label="Run 1", marker='o')
plt.plot(y2, label="Run 2", marker='s')
plt.ylabel("Training Loss (5-step avg)")
plt.title("Training Loss Comparison (5:1 Averaged)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
