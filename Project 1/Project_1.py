
import csv, argparse, numpy as np, time, datetime
from mpi4py import MPI

# Convert datetime string to timestamp
def parse_datetime(dt_str):
    return datetime.datetime.strptime(dt_str, '%m/%d/%Y %I:%M:%S %p').timestamp()

# Read CSV in chunks and distribute rows to processes
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
                    # Convert to numpy array for efficient sending
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
                    # Reset buffer and counter
                    buffer = []
                    counter = 0

            # Send remaining data in buffer at end of file
            if buffer:
                buffer_array = np.array(buffer, dtype=np.float64)
                if dest == rank:
                    append_data(buffer_array, x_train_local, y_train_local, x_test_local, y_test_local)
                else:
                    req = comm.isend(buffer_array, dest=dest, tag=0)
                    send_reqs.append(req)

            # Send end-of-data signal (None)
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

    # Convert list of arrays to single arrays and return to main
    X_train = np.vstack(x_train_local) if x_train_local else np.empty((0, len(features)-1))
    X_test = np.vstack(x_test_local) if x_test_local else np.empty((0, len(features)-1))
    # Add bias column (ones) to X_train and X_test
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    return (
        X_train,
        np.hstack(y_train_local) if y_train_local else np.empty(0),
        X_test,
        np.hstack(y_test_local) if y_test_local else np.empty(0)
    )

# Append data to local train/test sets
def append_data(buffer_array, x_train_local, y_train_local, x_test_local, y_test_local):
    
    # Split into x and y
    x = buffer_array[:, :-1]
    y = buffer_array[:, -1]

    # Random mask for train/test split
    mask = np.random.rand(len(x)) < 0.7

    x_train_local.extend(x[mask])
    y_train_local.extend(y[mask])
    x_test_local.extend(x[~mask])
    y_test_local.extend(y[~mask])

# Normalize training data to have zero mean and standard deviation of one
def normalize_data(x_train_local, x_test_local, comm):

    # Exclude last column (all 1 for bias) from normalization
    features = x_train_local.shape[1] - 1
    local_sum = x_train_local[:, :features].sum(axis=0)
    local_count = x_train_local.shape[0]

    global_sum = comm.allreduce(local_sum, op=MPI.SUM)
    global_count = comm.allreduce(local_count, op=MPI.SUM)

    mean = global_sum / global_count
    local_sq_diff = ((x_train_local[:, :features] - mean) ** 2).sum(axis=0)
    global_sq_diff = comm.allreduce(local_sq_diff, op=MPI.SUM)
    std = np.sqrt(global_sq_diff / global_count)

    # Normalize all but last column
    x_train_local[:, :features] -= mean
    x_train_local[:, :features] /= std
    x_test_local[:, :features] -= mean
    x_test_local[:, :features] /= std

    return x_train_local, x_test_local, mean, std

def initialize_weights(feature_count, neuron_count):
    hidden_weights = np.random.uniform(-0.1, 0.1, (feature_count + 1, neuron_count))
    output_weights = np.random.uniform(-0.1, 0.1, (neuron_count + 1, 1))

    return hidden_weights, output_weights

def compute_prediction(x, hidden_weights, output_weights, activation_id):

    # Matrix multiply x with hidden weights and apply activation function
    hidden_pre = x @ hidden_weights
    hidden_act = activation(hidden_pre, activation_id)
    # Add 1 for bias term
    hidden_act = np.hstack([hidden_act, np.ones((hidden_act.shape[0], 1))])
    # Matrix multiply activated hidden neuron outputs with output weights
    pred = hidden_act @ output_weights

    return hidden_pre, hidden_act, pred

def activation(x, id):
    if id == 0:
        return np.maximum(0, x) # ReLU
    elif id == 1:
        return 1 / (1 + np.exp(-x)) # Sigmoid
    elif id == 2:
        return np.tanh(x) # tanh
    else:
        raise ValueError("Unsupported activation function")
    
def activation_derivative(x, id):
    if id == 0:
        return (x > 0).astype(float) # ReLU
    elif id == 1:
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid * (1 - sigmoid) # Sigmoid
    elif id == 2:
        return 1 - np.tanh(x)**2 # tanh
    else:
        raise ValueError("Unsupported activation function")

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)
    args = parser.parse_args()
    filename = args.filename

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    neuron_count = 10
    feature_count = 9  # Number of features excluding bias

    np.random.seed(rank + int(time.time()))

    # Read and distribute data from csv
    x_train_local, y_train_local, x_test_local, y_test_local = read_csv(filename, comm, rank, size)
    
    # Normalize training data
    x_train_local, x_test_local, mean, std = normalize_data(x_train_local, x_test_local, comm)

    hidden_weights, output_weights = initialize_weights(feature_count, neuron_count)

if __name__ == "__main__":
    main()