
import pandas, csv, argparse, numpy as np, time, random, datetime
from mpi4py import MPI

def parse_datetime(dt_str):
    return datetime.datetime.strptime(dt_str, '%m/%d/%Y %I:%M:%S %p').timestamp()

def read_csv(filename, comm, rank, size):

    features = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count', 'trip_distance',
                'RatecodeID', 'PULocationID', 'DOLocationID', 'payment_type', 'extra', 'total_amount']

    chunk = 100
    max_reqs = 20
    send_reqs = []
    dest = 0

    x_train_local = []
    y_train_local = []
    x_test_local = []
    y_test_local = []

    if rank == 0:
        buffer = []
        counter = 0
        run_number = 0

        with open(filename, 'r') as csvfile:
            filereader = csv.reader(csvfile)
            header = next(filereader)
            column_index = [header.index(feature) for feature in features]

            for row in filereader:
                # Check for empty strings before conversion
                if (row[column_index[0]] == '' or row[column_index[1]] == '' or
                    any(row[idx] == '' for idx in column_index[2:])):
                    continue
                read_row = [parse_datetime(row[column_index[0]]),
                            parse_datetime(row[column_index[1]]),] + [float(row[idx]) for idx in column_index[2:]]
                buffer.append(read_row)
                counter += 1
                if counter == chunk:
                    run_number += 1
                    print(f"Rank {rank} processed chunk {run_number}")
                    buffer_array = np.array(buffer, dtype=np.float64)
                    if dest == rank:
                        append_data(buffer_array, x_train_local, y_train_local, x_test_local, y_test_local)
                    else:
                        if len(send_reqs) >= max_reqs:
                            MPI.Request.Waitall(send_reqs)
                            send_reqs = []
                        req = comm.isend(buffer_array, dest=dest, tag=0)
                        send_reqs.append(req)
                    dest = (dest + 1) % size
                    buffer = []
                    counter = 0

            if buffer:
                buffer_array = np.array(buffer, dtype=np.float32)
                if dest == rank:
                    append_data(buffer_array, x_train_local, y_train_local, x_test_local, y_test_local)
                else:
                    req = comm.isend(buffer_array, dest=dest, tag=0)
                    send_reqs.append(req)

            # Send end-of-data signal (None) to all non-root ranks
            for dest_rank in range(1, size):
                comm.send(None, dest=dest_rank, tag=0)

        if send_reqs:
            MPI.Request.Waitall(send_reqs)

    else:
        while True:
            buffer_array = comm.recv(source=0, tag=0)
            if buffer_array is None:
                break
            append_data(buffer_array, x_train_local, y_train_local, x_test_local, y_test_local)

    return x_train_local, y_train_local, x_test_local, y_test_local

def append_data(buffer, x_train_local, y_train_local, x_test_local, y_test_local):
    for row in buffer:
        if random.random() < 0.7:
            x_train_local.append(row[:-1])
            y_train_local.append(row[-1]) 
        else:
            x_test_local.append(row[:-1])
            y_test_local.append(row[-1])

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)
    args = parser.parse_args()
    filename = args.filename

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    random.seed(rank + time.time())

    x_train_local, y_train_local, x_test_local, y_test_local = read_csv(filename, comm, rank, size)
    # Write the local data to CSV files
    np.savetxt(f"x_train_local_rank_{rank}.csv", x_train_local, delimiter=",")
    np.savetxt(f"y_train_local_rank_{rank}.csv", y_train_local, delimiter=",")
    np.savetxt(f"x_test_local_rank_{rank}.csv", x_test_local, delimiter=",")
    np.savetxt(f"y_test_local_rank_{rank}.csv", y_test_local, delimiter=",")

if __name__ == "__main__":
    main()

print("Done")