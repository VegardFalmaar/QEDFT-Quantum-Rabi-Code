import time


class ProgressBar():
    def __init__(self, total: int, progress_bar_length: int = 30):
        self.total = total
        self.counter = 0
        self.start_time = int(time.perf_counter())
        self.pb_len = progress_bar_length

    def write(self, msg: str = ''):
        num_progress_bars = self.pb_len * self.counter // self.total
        progress = self.counter / self.total
        t = int(time.perf_counter()) - self.start_time
        total_t = int(t / progress) if self.counter > 0 else 0
        line = '[' + '#'*num_progress_bars \
            + ' '*(self.pb_len - num_progress_bars) \
            + f'] {progress*100:5.1f} % ' \
            + f'({t//60:3d}:{t%60:02d} / {total_t//60:3d}:{total_t%60:02d})'
        if msg != '':
            line += f'  ({msg})'
        print(line)
        self.counter += 1
