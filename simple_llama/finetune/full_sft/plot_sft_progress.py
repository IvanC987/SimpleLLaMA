import matplotlib.pyplot as plt


def average(lst, x):
    return [sum(lst[i:i+x]) / len(lst[i:i+x]) for i in range(0, len(lst), x)]


def clean_data(filepath, avg_val=None):
    with open(filepath, "r") as f:
        data = f.read().splitlines()[1:]

    training_loss = [float(i.split(",")[2]) for i in data]
    validation_loss = [float(i.split(",")[4]) for i in data]
    step = [int(i.split(",")[0]) for i in data]

    if avg_val is not None:
        training_loss = average(training_loss, avg_val)
        validation_loss = average(validation_loss, avg_val)
        step = average(step, avg_val)

    return training_loss, validation_loss, step



f1_path = "sft_progress.txt"
# f2_path = ""


files = [f1_path]

legends = files


colors = ['gold', 'lightgreen', 'deepskyblue', 'mediumblue', 'indigo', 'black']


x_avg = 1
training_loss1, validation_loss1, steps1 = clean_data(files[0], x_avg)
# training_loss2, validation_loss2, steps1 = clean_data(files[1], x_avg)

# --------------------------------------------------


plt.figure(figsize=(10, 6))

plt.plot(steps1, training_loss1, c=colors[0], label=legends[0] + "TL")
plt.plot(steps1, validation_loss1, c=colors[1], label=legends[0] + "VL")

# plt.plot(tokens_trained2, loss2, c=colors[1], label=legends[1])


plt.xlabel("Tokens Trained (Millions)", fontsize=12)
plt.ylabel("Training Loss", fontsize=12)
plt.title("Scaling Law Evaluation Across Model Sizes", fontsize=14)

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

# plt.xscale('log')
# plt.yscale('log')
plt.ylim(1.2, 3.0)

plt.legend(title="Model Size", fontsize=10)
plt.tight_layout()
plt.show()


