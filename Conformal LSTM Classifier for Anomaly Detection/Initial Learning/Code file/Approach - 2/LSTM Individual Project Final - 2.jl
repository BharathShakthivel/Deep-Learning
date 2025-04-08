#To clone the repo to the Local system
url = "https://github.com/numenta/NAB.git"
run(`git clone $url`)

# Libraries used
using Pkg
Pkg.add(["Flux", "CUDA", "CSV", "DataFrames", "Dates", "FFTW", "StatsBase",
         "Statistics", "Random", "Plots", "BSON","JSON","MLJTuning"])
using Flux, CUDA, BSON, CSV, DataFrames, Dates, Statistics, Random, Plots, JSON, BSON, MLJTuning, FFTW, StatsBase
using Flux: @epochs, train!

# Set random seed for reproducibility
Random.seed!(123)

# Function to parse the Timestamp
function parse_timestamps(timestamps)
    #Common formats
    formats = [
        dateformat"yyyy-mm-dd HH:MM:SS",
        dateformat"yyyy-mm-ddTHH:MM:SS",
        dateformat"yyyy-mm-ddTHH:MM:SS.s"
    ]
    
    for format in formats
        try
            return DateTime.(timestamps, format)
        catch
            continue
        end
    end
    
    # Error Handling
    # If all formats fail, print an example timestamp and raise error
    println("Example timestamp: $(timestamps[1])")
    error("Could not parse timestamps with any of the common formats")
end

# --- Dynamic Preprocessing ---

function adaptive_preprocessing(df)
    # Convert time differences to milliseconds as numeric values
    time_diffs = diff(df.timestamp)
    time_diffs_ms = Dates.value.(time_diffs)  # Convert to integer milliseconds
    
    # Calculate median frequency in milliseconds
    median_freq_ms = median(time_diffs_ms)
    median_freq = Millisecond(median_freq_ms)  # Convert back to Millisecond type
    
    # Auto-adjust window size based on frequency
    seq_len = if median_freq < Minute(5)
        128  # High frequency
    elseif median_freq < Hour(1)
        64
    else
        32
    end
    
    # Handle sparse anomalies
    if :label in names(df)
        anomaly_ratio = sum(df.label) / nrow(df)
        if anomaly_ratio < 0.01
            seq_len = min(seq_len, 32)
        end
    end
    
    # --- Data Variance Analysis ---
    data_values = df.value
    data_var = var(data_values)
    
    # Variance-based scaling (absolute thresholds)
    if data_var > 1e4
        seq_len = min(seq_len + 32, 256)
    elseif data_var < 1e2
        seq_len = max(seq_len - 16, 16)
    end
    
    return seq_len
end

# --- Data Preparation with Train-Test Split ---
function prepare_nab_data(file_path, test_size=0.2)
    # Load NAB data
    df = CSV.File(file_path) |> DataFrame
    
    # Convert timestamp to DateTime
    df.timestamp = parse_timestamps(df.timestamp)
    
    # Sort by timestamp
    sort!(df, :timestamp)
    
    # Calculate split point
    n = nrow(df)
    split_idx = floor(Int, n * (1 - test_size))
    
    # Split data
    train_df = df[1:split_idx, :]
    test_df = df[(split_idx+1):end, :]
    
    # Calculate normalization parameters from training data only
    train_mean = mean(train_df.value)
    train_std = std(train_df.value)
    
    # Normalize both sets using training parameters
    train_df.normalized_value = (train_df.value .- train_mean) ./ train_std
    test_df.normalized_value = (test_df.value .- train_mean) ./ train_std
    
    # Get dynamic sequence length from training data
    seq_len = adaptive_preprocessing(train_df)
    
    return train_df, test_df, train_mean, train_std, split_idx, seq_len
end

# --- Prepare sequences for model training and prediction ---
function create_sequences(data::Vector{Float64}, seq_len::Int)
    num_sequences = length(data) - seq_len
    
    X = Matrix{Float64}(undef, seq_len, num_sequences)
    Y = Vector{Float64}(undef, num_sequences)
    
    for i in 1:num_sequences
        range = i:(i+seq_len-1)
        X[:, i] = data[range]
        Y[i] = data[range[end]+1]
    end
    
    return reshape(X, 1, seq_len, :), Y
end


# --- Train model with Early Stopping ---
function train_lstm_model(X_train, Y_train, X_val, Y_val; 
    epochs=200, patience=20, min_delta=0.001)
    
    # Define model architecture
    model = Chain(
        LSTM(1 => 64),
        Dropout(0.2),
        xs -> xs[:, end, :],
        Dense(64 => 32),
        Dense(32 => 1),
        x -> vec(x)
    )
    
    # Optimizer
    opt = Adam(0.001)
    state = Flux.setup(opt, model)
    
    # For early stopping
    best_val_loss = Inf
    best_model = deepcopy(model)
    patience_counter = 0
    
    # Training history
    train_losses = Float32[]
    val_losses = Float32[]
    
    for epoch in 1:epochs
        # Reset LSTM state
        Flux.reset!(model)
        
        # Forward pass and gradient computation
        loss, grads = Flux.withgradient(model) do m
            preds = m(X_train)
            Flux.mae(preds, Y_train)
        end
        
        # Update model
        Flux.update!(state, model, grads)
        
        # Validate
        Flux.reset!(model)
        val_predictions = model(X_val)
        val_loss = Flux.mae(val_predictions, Y_val)
        
        # Store losses
        push!(train_losses, loss)
        push!(val_losses, val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss - min_delta
            best_val_loss = val_loss
            best_model = deepcopy(model)
            patience_counter = 0
        else
            patience_counter += 1
            if patience_counter >= patience
                println("Early stopping at epoch $epoch")
                break
            end
        end
        
        # Print progress
        if epoch % 10 == 0
            println("Epoch $epoch: Train Loss = $(round(loss, digits=4)), Val Loss = $(round(val_loss, digits=4))")
        end
    end
    
    # Plot training curves
    p = plot(
        train_losses, 
        label="Training Loss",
        linewidth=2,
        xlabel="Epoch",
        ylabel="Loss (MAE)",
        title="Training and Validation Loss"
    )
    plot!(p, val_losses, label="Validation Loss", linewidth=2, linestyle=:dash)
    display(p)
    
    return best_model, train_losses, val_losses
end

# --- Conformal Prediction Implementation ---
struct ConformalPredictor
    model::Chain
    calibration_errors::Vector{Float64}
    mean::Float64
    std::Float64
    seq_len::Int64 
end

function train_conformal_predictor(model, X_calib, Y_calib, mean_val, std_val, seq_len)
    # Reset LSTM state
    Flux.reset!(model)
    
    # Make predictions on calibration set
    predictions = model(X_calib)
    
    # Calculate absolute errors for calibration
    errors = abs.(predictions .- Y_calib)
    
    # Sort errors for quantile calculation
    sorted_errors = sort(errors)
    
    return ConformalPredictor(model, sorted_errors, mean_val, std_val, seq_len)
end

function predict_with_intervals(predictor::ConformalPredictor, X_new, alpha=0.1)
    # Reset LSTM state
    Flux.reset!(predictor.model)
    
    # Make point predictions
    point_predictions = predictor.model(X_new)
    
    # Get appropriate quantile from calibration errors
    n_calib = length(predictor.calibration_errors)
    quantile_idx = ceil(Int, (n_calib + 1) * (1 - alpha))
    
    if quantile_idx > n_calib
        quantile_idx = n_calib
    end
    
    # Get error margin for prediction interval
    error_margin = predictor.calibration_errors[quantile_idx]
    
    # Calculate prediction intervals
    lower_bound = point_predictions .- error_margin
    upper_bound = point_predictions .+ error_margin
    
    # Denormalize to original scale
    point_preds_original = (point_predictions .* predictor.std) .+ predictor.mean
    lower_bound_original = (lower_bound .* predictor.std) .+ predictor.mean
    upper_bound_original = (upper_bound .* predictor.std) .+ predictor.mean
    
    return point_preds_original, lower_bound_original, upper_bound_original
end

# --- Anomaly Detection with Conformal Prediction ---
function detect_anomalies(test_df, point_preds, lower_bound, upper_bound, seq_len)
    # Create a DataFrame for results
    results = DataFrame(
        timestamp = test_df.timestamp[seq_len+1:end],
        actual = test_df.value[seq_len+1:end],
        predicted = point_preds,
        lower_bound = lower_bound,
        upper_bound = upper_bound
    )
    
    # Identify anomalies (values outside the prediction interval)
    results.is_anomaly = (results.actual .< results.lower_bound) .| (results.actual .> results.upper_bound)
    
    # Calculate anomaly scores (distance from prediction, normalized by interval width)
    results.anomaly_score = abs.(results.actual .- results.predicted) ./ (results.upper_bound .- results.lower_bound)
    
    return results
end

# --- Convert results to NAB format ---
function convert_to_nab_format(results)
    # Format required by NAB: A csv with timestamp and anomaly likelihood
    nab_results = DataFrame(
        timestamp = string.(results.timestamp),
        value = results.anomaly_score
    )
    
    return nab_results
end

# --- Main Execution Pipeline ---
function run_conformal_anomaly_detection(file_path)
    # Prepare data
    dataset_name = basename(file_path)
    println("Processing dataset: $dataset_name")
    
    # Get dynamic sequence length from data
    train_df, test_df, train_mean, train_std, split_idx, seq_len = prepare_nab_data(file_path)
    
    # Further split training into train and validation
    n_train = nrow(train_df)
    val_size = 0.2
    val_split_idx = floor(Int, n_train * (1 - val_size))
    
    train_data = train_df[1:val_split_idx, :].normalized_value
    val_data = train_df[(val_split_idx+1):end, :].normalized_value
    
    # Prepare sequences
    seq_len = 64
    X_train, Y_train = create_sequences(train_data, seq_len)
    X_val, Y_val = create_sequences(val_data, seq_len)
    
    # Convert to Float32
    X_train, Y_train = Float32.(X_train), Float32.(Y_train)
    X_val, Y_val = Float32.(X_val), Float32.(Y_val)
    
    # Train model with early stopping
    println("Training LSTM model...")
    best_model, train_losses, val_losses = train_lstm_model(X_train, Y_train, X_val, Y_val)
    
    # Prepare test data
    X_test, Y_test = create_sequences(test_df.normalized_value, seq_len)
    X_test, Y_test = Float32.(X_test), Float32.(Y_test)
    
    # Create validation+calibration set for conformal prediction
    X_calib, Y_calib = X_val, Y_val
    
    # Train conformal predictor
    println("Training conformal predictor...")
    conformal_predictor = train_conformal_predictor(best_model, X_calib, Y_calib, train_mean, train_std,seq_len)
    
    # Make predictions with uncertainty
    println("Making predictions with uncertainty...")
    point_preds, lower_bound, upper_bound = predict_with_intervals(conformal_predictor, X_test, 0.1)
    
    # Detect anomalies
    println("Detecting anomalies...")
    anomaly_results = detect_anomalies(test_df, point_preds, lower_bound, upper_bound, seq_len)
    
    # Convert to NAB format
    nab_format_results = convert_to_nab_format(anomaly_results)
    
    # Save results for NAB scoring
    detection_file = joinpath("NAB", "outputs", "conformal_lstm", "realTraffic", "conformal_lstm_$(replace(dataset_name, ".csv" => "")).csv")
    mkpath(dirname(detection_file))
    CSV.write(detection_file, nab_format_results)
    

    # Print statistics
    n_anomalies = sum(anomaly_results.is_anomaly)
    anomaly_ratio = n_anomalies / length(anomaly_results.is_anomaly)
    println("Number of detected anomalies: $n_anomalies ($(round(anomaly_ratio * 100, digits=2))% of test data)")
    
    # Plot results
    plot_results(anomaly_results, dataset_name)
        
    # Save model
    save_model(conformal_predictor, dataset_name, seq_len)

    return conformal_predictor, anomaly_results, detection_file
end

# --- Visualization of results ---
function plot_results(results, dataset_name)
    # Plot the time series with prediction intervals and anomalies
    p = plot(
        results.timestamp, results.actual,
        label="Actual Values",
        linewidth=2,
        title="Anomaly Detection for $dataset_name",
        xlabel="Time",
        ylabel="Value",
        legend=:topright
    )
    
    plot!(p, results.timestamp, results.predicted, label="Predicted Values", linewidth=2, linestyle=:dash)
    
    # Plot prediction intervals as a ribbon
    plot!(
        p,
        results.timestamp,
        results.predicted,
        ribbon=(results.predicted .- results.lower_bound, results.upper_bound .- results.predicted),
        fillalpha=0.3,
        label="90% Prediction Interval"
    )
    
    # Highlight anomalies
    anomaly_points = results[results.is_anomaly, :]
    scatter!(
        p, 
        anomaly_points.timestamp, 
        anomaly_points.actual,
        markersize=6,
        color=:red,
        label="Detected Anomalies"
    )
    
    display(p)
    savefig(p, "anomaly_detection_$(replace(dataset_name, ".csv" => "")).png")
    
    # Plot anomaly scores
    p2 = plot(
        results.timestamp, 
        results.anomaly_score,
        linewidth=2,
        title="Anomaly Scores for $dataset_name",
        xlabel="Time",
        ylabel="Anomaly Score",
        legend=false
    )
    
    # Highlight points above threshold
    threshold = 1.0  # Adjust as needed
    high_scores = results[results.anomaly_score .> threshold, :]
    scatter!(
        p2,
        high_scores.timestamp,
        high_scores.anomaly_score,
        markersize=6,
        color=:red,
        label="High Anomaly Score"
    )
    
    display(p2)
    savefig(p2, "anomaly_scores_$(replace(dataset_name, ".csv" => "")).png")
end

# --- Complete NAB Benchmarking Pipeline ---
function run_nab_benchmark()
        # Use relative paths based on NAB repo structure
    datasets = [
        # IF any error comes with path use "NAB/data/realTraffic/TravelTime_387.csv"
        "data/realTraffic/TravelTime_387.csv"
    ]
    
    for dataset in datasets
        println("\n=== Processing: $dataset ===")
        full_path = joinpath("NAB/", dataset)
        run_conformal_anomaly_detection(full_path)
    end
end

# --- Saving the model ---
function save_model(predictor, dataset_name, seq_len)
    model_dir = "saved_models"
    mkpath(model_dir)
    BSON.bson(joinpath(model_dir, "$dataset_name.bson"), 
        Dict(
            :model => predictor.model,
            :calibration => predictor.calibration_errors,
            :stats => (predictor.mean, predictor.std),
            :seq_len => predictor.seq_len
        )
    )
end


# Training phase (automatically determines seq_len)
# Main Function to initiate the pipeline
run_nab_benchmark()

function predict_new_data(model_path::String, new_data_path::String)
    # 1. Load model with saved sequence length
    model_data = BSON.load(model_path)

    # Recreate the model structure
    model = Chain(
        LSTM(1 => 64),
        Dropout(0.2),
        xs -> xs[:, end, :],
        Dense(64 => 32),
        Dense(32 => 1),
        x -> vec(x)
    )

    predictor = ConformalPredictor(
        model,
        model_data[:calibration],
        model_data[:stats][1],  # mean
        model_data[:stats][2],  # std
        model_data[:seq_len]
    )
        
    # 2. Load and preprocess new data
    df = CSV.File(new_data_path) |> DataFrame
    
    # Parse timestamps
    df.timestamp = parse_timestamps(df.timestamp)
    sort!(df, :timestamp)
    
    # 3.1 Normalize using original training parameters
    df.normalized_value = (df.value .- predictor.mean) ./ predictor.std
    
    #3.2 Adaptive_preprocessing
    seq_len = adaptive_preprocessing(df)

    # 4. Create sequences for LSTM
    X_new, Y_new = create_sequences(df.normalized_value, seq_len)
    X_new = Float32.(X_new)  # Match training precision
    
    # 5. Generate predictions with uncertainty
    point_preds, lower, upper = predict_with_intervals(predictor, X_new, 0.1)
    
    # 6. Calculate anomaly scores (even without labels)
    results = DataFrame(
        timestamp = df.timestamp[seq_len+1:end],
        value = df.value[seq_len+1:end],
        predicted = point_preds,
        anomaly_score = abs.(df.value[seq_len+1:end] .- point_preds) ./ (upper .- lower)
    )
    
    # 7. Threshold anomalies (adjust threshold as needed)
    results.is_anomaly = results.anomaly_score .> 3.0
    
    return results
end


# Prediction phase (uses stored seq_len)
new_results = predict_new_data(
    "saved_models/TravelTime_387.csv.bson", # Loading our k_fold_cross_validation model
    "NAB/data/realTraffic/TravelTime_451.csv"
)

# Save results
CSV.write("new_anomaly_predictions.csv", new_results)

dataset_name = "Travel_time_451"

# Plotting results

# Visualize
p1 = plot(
    new_results.timestamp, 
    new_results.anomaly_score,
    title="Anomaly Scores for Travel_Time 451 Data",
    label="Score", 
    linewidth=2
)
hline!([5.0], label="Threshold", linestyle=:dash, linecolor=:red)

savefig(p1, "anomaly_scores_$(replace(dataset_name, ".csv" => "")).png")


# Plot anomaly scores
p2 = plot(
        new_results.timestamp, 
        new_results.anomaly_score,
        linewidth=2,
        title="Anomaly Scores for $dataset_name",
        xlabel="Time",
        ylabel="Anomaly Score",
        legend=false
    )
    
    # Highlight points above threshold
threshold = 5.5  # Adjust as needed
high_scores = new_results[new_results.anomaly_score .> threshold, :]
    scatter!(
        p2,
        high_scores.timestamp,
        high_scores.anomaly_score,
        markersize=6,
        color=:red,
        label="High Anomaly Score"
    )
    
display(p2)
savefig(p2, "anomaly_scores_$(replace(dataset_name, ".csv" => "")).png")

# Using the model which was trained for Traffic_Travel-time – 387 to predict 
# the conformal scores for the data Travel_time- 451 and compare the results with NAB benchmark


# Load CSV file into a DataFrame
function load_file(file_path)
    return CSV.File(file_path) |> DataFrame
end

# Align predictions with NAB results based on timestamp
function align_results(nab_results, my_results)
    # Ensure timestamp format is consistent
    if !(eltype(nab_results.timestamp) <: DateTime)
        nab_results.timestamp = DateTime.(nab_results.timestamp, dateformat"yyyy-mm-dd HH:MM:SS")
    end
    if !(eltype(my_results.timestamp) <: DateTime)
        my_results.timestamp = DateTime.(my_results.timestamp, dateformat"yyyy-mm-ddTHH:MM:SS")
    end

    # Create new column in NAB data to hold predicted values (default 0)
    nab_results.predicted_value = zeros(Float64, nrow(nab_results))

    # Replace predicted_value at matching timestamps from my_results
    for row in eachrow(my_results)
        idx = findfirst(x -> x == row.timestamp, nab_results.timestamp)
        if idx !== nothing
            nab_results.predicted_value[idx] = row.anomaly_score
        end
    end

    return nab_results
end

# Calculates binary classification metrics using a threshold
function calculate_regression_metrics(predicted_scores, true_scores)
    errors = predicted_scores .- true_scores
    mse = mean(errors .^ 2)
    rmse = sqrt(mse)
    mae = mean(abs.(errors))

    return rmse, mae, mse
end


# Load both files
nab_results = load_file("NAB/results/numenta/realTraffic/numenta_TravelTime_451.csv")
my_results = load_file("new_anomaly_predictions.csv")

#Normalising the values to compare the results
my_results.anomaly_score = (my_results.anomaly_score .- minimum(my_results.anomaly_score))/
(maximum(my_results.anomaly_score) .- minimum(my_results.anomaly_score))


# Align your model predictions with NAB results
aligned_results = align_results(nab_results, my_results)
CSV.write("aligned_results_predictions.csv", aligned_results)
# Results
# From NAB: actual anomaly scores
true_scores = aligned_results.anomaly_score

# From model: predicted anomaly scores
predicted_scores = aligned_results.predicted_value

calculate_regression_metrics(predicted_scores, true_scores)

# Print the results

rmse, mae, mse = calculate_regression_metrics(predicted_scores, true_scores)

println("RMSE: $rmse")
println("MAE: $mae")
println("MSE: $mse")
