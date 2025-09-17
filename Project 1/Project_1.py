
import csv, argparse, numpy as np, time, datetime, matplotlib.pyplot as plt
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

    chunk = 100
    max_reqs = 20
    send_reqs = []
    dest = 0

    if rank == 0:
        buffer_train = []
        buffer_test = []
        counter_train = 0
        counter_test = 0
        with open(filename, 'r') as csvfile:
            filereader = csv.reader(csvfile)
            header = next(filereader)
            column_index = [header.index(feature) for feature in features]
            row_idx = 0
            for row in filereader:
                if (row[column_index[0]] == '' or row[column_index[1]] == '' or
                    any(row[idx] == '' for idx in column_index[2:])):
                    continue
                read_row = [parse_datetime(row[column_index[0]]),
                            parse_datetime(row[column_index[1]]),] + [float(row[idx]) for idx in column_index[2:]]
                # Deterministic split: use row_idx for reproducibility
                if row_idx % 10 < 7:  # 70% train, 30% test
                    buffer_train.append(read_row)
                    counter_train += 1
                    if counter_train == chunk:
                        buffer_array = np.array(buffer_train, dtype=np.float64)
                        if dest == rank:
                            append_data(buffer_array, x_train_local, y_train_local, x_test_local, y_test_local, is_train=True)
                        else:
                            if len(send_reqs) >= max_reqs:
                                MPI.Request.Waitall(send_reqs)
                                send_reqs = []
                            req = comm.isend(buffer_array, dest=dest, tag=1)
                            send_reqs.append(req)
                        dest = (dest + 1) % size
                        buffer_train = []
                        counter_train = 0
                else:
                    buffer_test.append(read_row)
                    counter_test += 1
                    if counter_test == chunk:
                        buffer_array = np.array(buffer_test, dtype=np.float64)
                        if dest == rank:
                            append_data(buffer_array, x_train_local, y_train_local, x_test_local, y_test_local, is_train=False)
                        else:
                            if len(send_reqs) >= max_reqs:
                                MPI.Request.Waitall(send_reqs)
                                send_reqs = []
                            req = comm.isend(buffer_array, dest=dest, tag=2)
                            send_reqs.append(req)
                        dest = (dest + 1) % size
                        buffer_test = []
                        counter_test = 0
                row_idx += 1
        # Send remaining data in buffers
        if buffer_train:
            buffer_array = np.array(buffer_train, dtype=np.float64)
            if dest == rank:
                append_data(buffer_array, x_train_local, y_train_local, x_test_local, y_test_local, is_train=True)
            else:
                req = comm.isend(buffer_array, dest=dest, tag=1)
                send_reqs.append(req)
        if buffer_test:
            buffer_array = np.array(buffer_test, dtype=np.float64)
            if dest == rank:
                append_data(buffer_array, x_train_local, y_train_local, x_test_local, y_test_local, is_train=False)
            else:
                req = comm.isend(buffer_array, dest=dest, tag=2)
                send_reqs.append(req)
        # Send end-of-data signal (None) for both train and test
        for dest_rank in range(1, size):
            comm.send(None, dest=dest_rank, tag=1)
            comm.send(None, dest=dest_rank, tag=2)
        if send_reqs:
            MPI.Request.Waitall(send_reqs)
    else:
        while True:
            buffer_array = comm.recv(source=0, tag=1)
            if buffer_array is None:
                break
            append_data(buffer_array, x_train_local, y_train_local, x_test_local, y_test_local, is_train=True)
        while True:
            buffer_array = comm.recv(source=0, tag=2)
            if buffer_array is None:
                break
            append_data(buffer_array, x_train_local, y_train_local, x_test_local, y_test_local, is_train=False)

    # Convert lists to arrays and add bias
    X_train = np.vstack(x_train_local) if x_train_local else np.empty((0, len(features)-1))
    X_test = np.vstack(x_test_local) if x_test_local else np.empty((0, len(features)-1))
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    y_train = np.hstack(y_train_local) if y_train_local else np.empty(0)
    y_test = np.hstack(y_test_local) if y_test_local else np.empty(0)
    return X_train, y_train, X_test, y_test

# Append data to local train/test sets
def append_data(buffer_array, x_train_local, y_train_local, x_test_local, y_test_local, is_train):
    x = buffer_array[:, :-1]
    y = buffer_array[:, -1]
    if is_train:
        x_train_local.extend(x)
        y_train_local.extend(y)
    else:
        x_test_local.extend(x)
        y_test_local.extend(y)

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

    if comm.rank == 0:
        print("Mean used:", mean)
        print("Std used:", std)

    return x_train_local, x_test_local

# Initialize weights and broadcast to all processes
def initialize_weights(feature_count, neuron_count, comm, rank):

    if rank == 0:
        # Initialize weights randomly between -0.1 and 0.1
        hidden_weights = np.random.uniform(-0.1, 0.1, (feature_count + 1, neuron_count))
        output_weights = np.random.uniform(-0.1, 0.1, (neuron_count + 1, 1))

    else:
        # Declare empty arrays to receive weights from root
        hidden_weights = np.empty((feature_count + 1, neuron_count), dtype=np.float64)
        output_weights = np.empty((neuron_count + 1, 1), dtype=np.float64)

    # Broadcast weights from root
    reqs = []
    reqs.append(comm.Ibcast(hidden_weights, root=0))
    reqs.append(comm.Ibcast(output_weights, root=0))
    MPI.Request.Waitall(reqs)

    return hidden_weights, output_weights

# Forward pass to compute predictions
def compute_prediction(x, hidden_weights, output_weights, activation_id):

    # Matrix multiply x with hidden weights and apply activation function
    hidden_pre = x @ hidden_weights 
    hidden_act = activation(hidden_pre, activation_id)
    # Add 1 for bias term
    hidden_act = np.hstack([hidden_act, np.ones((hidden_act.shape[0], 1))])
    # Matrix multiply activated hidden neuron outputs with output weights
    pred = hidden_act @ output_weights

    return hidden_pre, hidden_act, pred

# Activation functions based on user selection
def activation(x, id):

    if id == 0:
        return np.maximum(0, x) # ReLU
    elif id == 1:
        return 1 / (1 + np.exp(-x)) # Sigmoid
    elif id == 2:
        return np.tanh(x) # tanh
    else:
        raise ValueError("Invalid actvation function selection")
    
# Derivatives of activation functions
def activation_derivative(x, id):

    if id == 0:
        return (x > 0).astype(float) # ReLU
    elif id == 1:
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid * (1 - sigmoid) # Sigmoid
    elif id == 2:
        return 1 - np.tanh(x)**2 # tanh
    else:
        raise ValueError("Invalid actvation function selection")
    
# Backpropagation step to compute gradient
def compute_gradient(x, y, hidden_weights, output_weights, activation_id):
    
    batch_size = x.shape[0]

    # Forward pass
    hidden_pre, hidden_act, pred = compute_prediction(x, hidden_weights, output_weights, activation_id)

    error = pred.flatten() - y

    # Backpropagation
    grad_output = hidden_act.T @ error[:, None] / batch_size     # σ * error
    act_deriv = activation_derivative(hidden_pre, activation_id)        # σ'
    delta_hidden = (error[:, None] @ output_weights[:-1].T) * act_deriv     # error * w_j * σ'
    grad_hidden = x.T @ delta_hidden / batch_size       # x * (error * w_j * σ')

    return grad_hidden, grad_output

# Update weights using averaged gradients from all processes
def update_weights(comm, hidden_weights, output_weights, grad_hidden, grad_output, learning_rate):

    # Get local gradients from all processes and calculate average
    grad_hidden_global = comm.allreduce(grad_hidden, op=MPI.SUM) / comm.Get_size()
    grad_output_global = comm.allreduce(grad_output, op=MPI.SUM) / comm.Get_size()

    # Update weights using global gradients
    hidden_weights -= learning_rate * grad_hidden_global
    output_weights -= learning_rate * grad_output_global

    return hidden_weights, output_weights

# Compute SSE, helper function for RMSE and loss
def compute_sse(x, y, hidden_weights, output_weights, activation_id, comm):

    # Compute f(x) for SSE
    _, _, pred = compute_prediction(x, hidden_weights, output_weights, activation_id)

    # Compute local SSE
    local_sse = np.sum((pred.flatten() - y) ** 2)
    local_count = len(y)


    global_sse = comm.allreduce(local_sse, op=MPI.SUM)
    global_count = comm.allreduce(local_count, op=MPI.SUM)

    return global_sse, global_count

# Compute RMSE for array x
def compute_rmse(x, y, hidden_weights, output_weights, activation_id, comm):

    # Get global SSE and row count
    global_sse, global_count = compute_sse(x, y, hidden_weights, output_weights, activation_id, comm)

    # Return RMSE
    return np.sqrt(global_sse / global_count)

# Compute training loss for array x
def compute_loss(x, y, hidden_weights, output_weights, activation_id, comm):

    # Get global SSE and row count
    global_sse, global_count = compute_sse(x, y, hidden_weights, output_weights, activation_id, comm)

    # Return loss
    return 0.5 * (global_sse / global_count)

# Train model using mini-batches
def train_model(x, y, hidden_weights, output_weights, activation_id, comm, learning_rate,
                stopping_criterion, max_iterations, M):
    
    # Initialize stopping criteria trackers
    loss_delta = 1e6
    previous_loss = compute_loss(x, y, hidden_weights, output_weights, activation_id, comm)
    iteration = 0

    # Initialize loss history
    loss_history = []

    # Run gradient descent training loop until stopping criteria met
    while loss_delta > stopping_criterion and iteration < max_iterations:

        # Randomly select M rows from local training data for mini-batch
        index = np.random.choice(x.shape[0], M, replace=False)
        x_batch = x[index]
        y_batch = y[index]

        # Compute gradients and update weights
        grad_hidden, grad_output = compute_gradient(x_batch, y_batch, hidden_weights, output_weights, activation_id)
        hidden_weights, output_weights = update_weights(comm, hidden_weights, output_weights, grad_hidden, grad_output, learning_rate)
       
        # Compute current loss and loss delta
        current_loss = compute_loss(x_batch, y_batch, hidden_weights, output_weights, activation_id, comm)
        loss_delta = abs(previous_loss - current_loss)

        # Set previous loss for next iteration
        previous_loss = current_loss
        iteration += 1

        if comm.Get_rank() == 0:
            print(f"Iteration {iteration}, Loss: {current_loss}, Loss Delta: {loss_delta}")
            loss_history.append(current_loss)


    return hidden_weights, output_weights, loss_history

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)
    args = parser.parse_args()
    filename = args.filename

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    neuron_count = 32   # Number of neurons in hidden layer
    feature_count = 9   # Number of features excluding bias
    M = 500   # Batch size
    learning_rate = 0.00001
    activation_id = 0   # 0: ReLU, 1: Sigmoid, 2: tanh
    stopping_criterion = 1e-5
    max_iterations = 10000

    np.random.seed(rank + int(time.time()))

    # Read and distribute data from csv
    x_train_local, y_train_local, x_test_local, y_test_local = read_csv(filename, comm, rank, size)
    
    # Normalize training data
    x_train_local, x_test_local = normalize_data(x_train_local, x_test_local, comm)

   # Initialize weights
    hidden_weights, output_weights = initialize_weights(feature_count, neuron_count, comm, rank)

#    hidden_weights, output_weights, loss_history = train_model(x_train_local, y_train_local, hidden_weights, output_weights,
#                                                    activation_id, comm, learning_rate,
#                                                    stopping_criterion, max_iterations, M)
    
    # Compute and print RMSE on training and test data
#    train_rmse = compute_rmse(x_train_local, y_train_local, hidden_weights, output_weights, activation_id, comm)
#    test_rmse = compute_rmse(x_test_local, y_test_local, hidden_weights, output_weights, activation_id, comm)
#    _, _, predictions = compute_prediction(x_train_local, hidden_weights, output_weights, activation_id)

#    print(predictions.flatten())
#    print(y_train_local)

#    if rank == 0:
#        print(f"Final Training RMSE: {train_rmse}")
#        print(f"Final Test RMSE: {test_rmse}")
    
    if rank == 0:
        
        mean_x_train = np.mean(x_train_local, axis=0)
        mean_x_test = np.mean(x_test_local, axis=0)
        mean_y_train = np.mean(y_train_local)
        mean_y_test = np.mean(y_test_local)

        print(f"Mean of training feature values: {mean_x_train}")
        print(f"Mean of test feature values: {mean_x_test}")
        print(f"Mean of training target values: {mean_y_train}")
        print(f"Mean of test target values: {mean_y_test}")

if __name__ == "__main__":
    main()