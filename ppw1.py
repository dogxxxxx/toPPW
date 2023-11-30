import time
import sys
from itertools import product

import numpy as np
import pandas as pd


start = time.time()


def matrix_computing(target: int, d: dict):
    arrays = []
    columns = []
    for k, v in d.items():
        arrays.append(np.arange(v + 1))
        columns.append(k)

    mesh = np.meshgrid(*arrays)
    values = [x[y] for x, y in zip(arrays, mesh)]
    sums = sum(values)

    filter_idx = np.where(sums == target)

    combo_vals = np.vstack([x[filter_idx] for x in values]).T

    df = pd.DataFrame(combo_vals, columns=columns)
    return df


def main(d: dict, target: int, for_loop_amount: int = 2):
    max_n_dict = {}
    for _ in range(for_loop_amount):
        key = max(d, key=d.get)
        val = d.pop(key)
        max_n_dict[key] = val

    loops = [range(value+1) for value in max_n_dict.values()]
    columns = list(max_n_dict.keys())
    d_sum = sum(d.values())

    ind = 0

    whole_df = pd.DataFrame()
    for combo in product(*loops):
        para_target = target - sum(combo)
        if para_target < 0 or para_target > d_sum:
            continue
        df1 = matrix_computing(para_target, d)
        df2 = pd.DataFrame([combo] * len(df1), columns=columns)
        merged_df = pd.concat([df1, df2], axis=1)
        whole_df = pd.concat([whole_df, merged_df], ignore_index=True)

        if ind % 500 == 0 and ind != 0:
            whole_df.to_csv(f'df{ind}.csv', index=False)
            whole_df = pd.DataFrame()
        ind += 1

    whole_df.to_csv(f'df{ind}.csv', index=False)


if __name__ == "__main__":
    start = time.time()
    data = dict()
    target = int(sys.argv[1])
    for arg in sys.argv[2:]:
        key, value = arg.split("=")
        value = int(value)
        data[key] = value

    main(data, target, 2)
    print(f'time consumed: {time.time() - start} seconds')
