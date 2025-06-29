import matplotlib.pyplot as plt
import os



def average(lst, x):
    return [sum(lst[i:i+x]) / len(lst[i:i+x]) for i in range(0, len(lst), x)]


def clean_data(filepath, avg_val=None):
    with open(filepath, "r") as f:
        data = f.read().splitlines()[1:]

    loss = [float(i.split(",")[2]) for i in data]
    tokens_trained = [int(i.split(",")[7]) for i in data]

    if avg_val is not None:
        loss = average(loss, avg_val)
        tokens_trained = average(tokens_trained, avg_val)

    return loss, tokens_trained





folders = [name for name in os.listdir('.') if os.path.isdir(os.path.join('.', name))]
folders = sorted(folders, key=lambda x: float(x.split("M")[0]))
files = [os.path.join(name, "training_progress.txt") for name in folders]

legends = [name.split("-")[0] for name in folders]


colors = ['gold', 'lightgreen', 'deepskyblue', 'mediumblue', 'indigo', 'black']


x_avg = 5
loss1, tokens_trained1 = clean_data(files[0], x_avg)
loss2, tokens_trained2 = clean_data(files[1], x_avg)
loss3, tokens_trained3 = clean_data(files[2], x_avg)
loss4, tokens_trained4 = clean_data(files[3], x_avg)
loss5, tokens_trained5 = clean_data(files[4], x_avg)
loss6, tokens_trained6 = clean_data(files[5], x_avg)

# --------------------------------------------------


plt.figure(figsize=(10, 6))

plt.plot(tokens_trained1, loss1, c=colors[0], label=legends[0])
plt.plot(tokens_trained2, loss2, c=colors[1], label=legends[1])
plt.plot(tokens_trained3, loss3, c=colors[2], label=legends[2])
plt.plot(tokens_trained4, loss4, c=colors[3], label=legends[3])
plt.plot(tokens_trained5, loss5, c=colors[4], label=legends[4])
plt.plot(tokens_trained6, loss6, c=colors[5], label=legends[5])

plt.xlabel("Tokens Trained (Millions)", fontsize=12)
plt.ylabel("Training Loss", fontsize=12)
plt.title("Scaling Law Evaluation Across Model Sizes", fontsize=14)

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

# plt.xscale('log')
# plt.yscale('log')
plt.ylim(2.3, 4)

plt.legend(title="Model Size", fontsize=10)
plt.tight_layout()
plt.show()

