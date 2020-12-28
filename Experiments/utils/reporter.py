import csv

class Reporter:
    def __init__(self, filename):
        self.filename = filename
        header = ['operation', 'epoch', 'batch', 'loss', 'accuracy', 'memory', 'time']
        with open(self.filename, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(header)

    def report(self, operation=None, epoch=None, batch=None, loss=None, accuracy=None, memory=None, time=None):
        data = [operation, epoch, batch, loss, accuracy, memory, time]
        with open(self.filename, 'a+') as file:
            writer = csv.writer(file)
            writer.writerow(data)
