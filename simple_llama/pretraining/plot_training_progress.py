import matplotlib.pyplot as plt


def parse_training_loss(lines):
    header = lines[0].split(",")
    data = lines[1:]

    token_idx = header.index("Tokens Processed (Total- In Millions)")
    loss_idx = header.index("Training Loss")

    x = []
    y = []

    for line in data:
        parts = line.split(",")
        tokens = float(parts[token_idx]) / 1000  # Convert to billions
        loss = float(parts[loss_idx])
        x.append(tokens)
        y.append(loss)

    return x, y


def parse_l2_norm(lines):
    header = lines[0].split(",")
    data = lines[1:]

    token_idx = header.index("Tokens Processed (Total- In Millions)")
    norm_idx = header.index("L2 Norm")

    x = []
    y = []

    for line in data:
        parts = line.split(",")
        tokens = float(parts[token_idx]) / 1000  # Convert to billions
        norm = float(parts[norm_idx])
        x.append(tokens)
        y.append(norm)

    return x, y


if __name__ == "__main__":
    with open("training_progress.txt", "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    # Parse plots
    x_loss, y_loss = parse_training_loss(lines)
    x_norm, y_norm = parse_l2_norm(lines)

    # Plot Training Loss
    plt.figure(figsize=(10, 6))
    plt.plot(x_loss, y_loss, label="Training Loss")
    plt.xlabel("Tokens Trained (Billions)")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs Tokens Trained")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot L2 Norm
    plt.figure(figsize=(10, 6))
    plt.plot(x_norm, y_norm, label="L2 Norm", color="orange")
    plt.xlabel("Tokens Trained (Billions)")
    plt.ylabel("L2 Norm")
    plt.title("L2 Norm vs Tokens Trained")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Compute total training time
    time_idx = lines[0].split(",").index("Time Per Evaluation")
    total_seconds = sum(int(line.split(",")[time_idx]) for line in lines[1:])
    print(f"Total Training Time: {total_seconds / 60:.2f} minutes ({total_seconds / 3600:.2f} hours)")
