import os
import time

last_data = []

data_path = "/home/stefan/Documents/HTEC/MLOps/data"
while True:
    new_data = [x for x in os.listdir(data_path) if x not in last_data]
    if len(new_data) != 0:
        print(f"New data arrived! {new_data[0]}")
        os.system(f"python train.py --data_path={os.path.join(data_path, new_data[0])}")
        last_data.append(new_data[0])
    else:
        print("No new data! ")
    time.sleep(3)
