import matplotlib.pyplot as plt
import csv

train_box_loss = []
train_cls_loss = []
val_box_loss = []
val_cls_loss = []

with open('../results_best_model.csv', newline='') as csvfile:
    data = csv.reader(csvfile, delimiter=' ', quotechar='|')

    HEADERS = ["epoch", "time", "train/box_loss", "train/cls_loss", "train/dfl_loss", "metrics/precision(B)",
               "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)", "val/box_loss", "val/cls_loss",
               "val/dfl_loss", "lr/pg0", "lr/pg1", "lr/pg2"]

    for raw_row in data:
        row = raw_row[0].split(',')
        train_box_loss.append(float(row[2]))
        train_cls_loss.append(float(row[3]))
        val_box_loss.append(float(row[9]))
        val_cls_loss.append(float(row[10]))

plt.figure()
plt.plot(train_box_loss, label='train')
plt.plot(val_box_loss, label='validation')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title("Box loss Train/Validation")
plt.legend()
plt.show()

plt.figure()
plt.plot(train_cls_loss, label='train')
plt.plot(val_cls_loss, label='validation')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title("Class loss Train/Validation")
plt.legend()
plt.show()
