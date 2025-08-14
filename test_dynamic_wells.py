import pandas as pd

def set_constant_rt_every_100_rows(depth_col='DEPTH', rt_col='RT'):
    """
    Set RT to a value that changes every 100 rows:
    1, 2, ..., 9, 10, 20, ..., 90, 100, 200, ...
    """
    df = pd.read_csv('data/structures/adera/benuang/BNG-057.csv')
    # Calculate interval by row index
    interval = (df.index // 100).astype(int)
    # Generate RT value for each interval
    def rt_value(n):
        if n < 10:
            return n + 1
        else:
            power = 10 ** (len(str(n)) - 1)
            return (n // power) * power
    df[rt_col] = interval.apply(rt_value)
    df.to_csv('BNG-057-RT-TEST.csv', index=False)
    return True

set_constant_rt_every_100_rows()