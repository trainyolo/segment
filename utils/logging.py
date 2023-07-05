import os

class Logger:

    def __init__(self, keys, title=""):

        self.data = {k: [] for k in keys}
        self.title = title

        print('created logger with keys:  {}'.format(keys))

    def save(self, save_dir="./output"):
            # save data as csv
            with open(os.path.join(save_dir, self.title + '.csv'), 'w') as file:
                # write header
                file.write(','.join(self.data.keys()) + '\n')
                
                # write data
                for item in zip(*self.data.values()):
                    item = [f'{i:.5f}' for i in item]
                    file.write(','.join(item) + '\n')

    def add(self, key, value):
        assert key in self.data, "Key not in data"
        self.data[key].append(value)