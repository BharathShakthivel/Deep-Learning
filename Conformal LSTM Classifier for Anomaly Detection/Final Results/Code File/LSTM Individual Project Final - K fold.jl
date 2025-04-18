"Real time traffic data from the Twin Cities Metro area in Minnesota, collected by the Minnesota Department of Transportation. 
Included metrics include occupancy, speed, and travel time from specific sensors."


# Libraries
using Pkg
Pkg.add(["CSV", "DataFrames", "Flux", "Plots", "Statistics", "Random","ConformalPrediction","MLJ","StatsBase","BSON"])
using Flux          
using Flux.Optimise 
using Statistics    
using Plots         
using Random        
using CSV           
using DataFrames    
using Flux.Optimise: Adam  
using ConformalPrediction
using MLJ
using BSON
using BSON: @save, @load
using StatsBase


# --- Data Preparation ---
data = CSV.File("NAB/data/realTraffic/TravelTime_387.csv"; types=Dict(:value => Float64)) |> DataFrame
data.value = (data.value .- mean(data.value)) ./ std(data.value)



# --- Function for Sequence Creation ---
function create_sequences(data, seq_len::Int)
    # Handle both DataFrame and Vector inputs
    values = data isa DataFrame ? data.value : data
    
    num_sequences = length(values) - seq_len
    
    X = Matrix{Float64}(undef, seq_len, num_sequences)
    Y = Vector{Float64}(undef, num_sequences)
    
    for i in 1:num_sequences
        range = i:(i+seq_len-1)
        X[:, i] = values[range]
        Y[i] = values[range[end]+1]
    end
    
    return reshape(X, 1, seq_len, :), Y
end
# # Sequence creation
seq_len = 64
X, Y = create_sequences(data, seq_len)

function safe_gradient_clipping(grads, max_norm=1.0f0)
    # Flatten gradients safely for different layer types
    function extract_grad_params(g)
        if g isa NamedTuple
            # Handle different possible gradient structures
            grad_params = []
            for field in propertynames(g)
                param = getproperty(g, field)
                if param isa AbstractArray
                    push!(grad_params, vec(param))
                elseif param isa NamedTuple
                    # Handle nested named tuples (like for some layer types)
                    for nested_field in propertynames(param)
                        nested_param = getproperty(param, nested_field)
                        if nested_param isa AbstractArray
                            push!(grad_params, vec(nested_param))
                        end
                    end
                end
            end
            return vcat(grad_params...)
        elseif g isa AbstractArray
            return vec(g)
        else
            return Float32[]
        end
    end
    
    # Safely extract and flatten all gradients
    flat_grads = vcat([extract_grad_params(g) for g in grads]...)
    
    # Prevent empty gradient vector
    if isempty(flat_grads)
        return grads
    end
    
    # Compute total gradient norm
    total_norm = sqrt(sum(x -> x^2, flat_grads))
    
    # Compute scaling factor
    scale = min(max_norm / (total_norm + 1f-6), 1.0f0)
    
    # Scale gradients while preserving original structure
    scaled_grads = map(grads) do g
        if g isa NamedTuple
            # Reconstruct named tuple with scaled parameters
            scaled_params = map(propertynames(g)) do field
                param = getproperty(g, field)
                if param isa AbstractArray
                    return param .* scale
                elseif param isa NamedTuple
                    # Handle nested named tuples (like for some layer types)
                    scaled_nested = map(nested_field -> 
                        getproperty(param, nested_field) isa AbstractArray ? 
                        getproperty(param, nested_field) .* scale : 
                        getproperty(param, nested_field), 
                        propertynames(param)
                    )
                    return NamedTuple{propertynames(param)}(scaled_nested)
                else
                    return param
                end
            end
            return NamedTuple{propertynames(g)}(scaled_params)
        elseif g isa AbstractArray
            return g .* scale
        else
            return g
        end
    end
    
    return scaled_grads
end

# We can go with the K Fold cross validation approach to obtain the best model.

# Cross-Validation Functions

function create_k_fold_splits(data::Vector{Float64}, k::Int, seq_len::Int)
    # Determine the size of each fold
    fold_size = div(length(data), k)
    
    # Prepare to store k different train/validation sets
    train_sets = Vector{Tuple{Array{Float32, 3}, Vector{Float32}}}(undef, k)
    val_sets = Vector{Tuple{Array{Float32, 3}, Vector{Float32}}}(undef, k)
    
    for i in 1:k
        # Create a copy of the data to split
        fold_data = copy(data)
        
        # Determine validation range
        val_start = (i-1)*fold_size + 1
        val_end = min(i*fold_size, length(data))
        
        # Remove validation data from training data
        validation_slice = fold_data[val_start:val_end]
        fold_data[val_start:val_end] .= NaN
        fold_data = fold_data[.!isnan.(fold_data)]
        
        # Normalize training data
        train_mean = mean(fold_data)
        train_std = std(fold_data)
        normalized_train = (fold_data .- train_mean) ./ train_std
        normalized_val = (validation_slice .- train_mean) ./ train_std
        
        # Create sequences for training
        X_train, Y_train = create_sequences(normalized_train, seq_len)
        X_train = Float32.(X_train)
        Y_train = Float32.(Y_train)
        
        # Create sequences for validation
        X_val, Y_val = create_sequences(normalized_val, seq_len)
        X_val = Float32.(X_val)
        Y_val = Float32.(Y_val)
        
        train_sets[i] = (X_train, Y_train)
        val_sets[i] = (X_val, Y_val)
    end
    
    return train_sets, val_sets
end

function train_and_evaluate_model(X_train, Y_train, X_val, Y_val; epochs=100)
    # Define model architecture
    model_cv = Chain(
        LSTM(1 => 64),
        Dropout(0.2),
        xs -> xs[:, end, :],
        Dense(64 => 32),
        Dense(32 => 1),
        x -> vec(x)
    )
    
    # Optimizer
    opt = Adam(0.001)
    state = Flux.setup(opt, model_cv)
    
    # Training loop
    for epoch in 1:epochs
        Flux.reset!(model_cv)
        
        loss, grads = Flux.withgradient(model_cv) do m
            preds = m(X_train)
            Flux.mae(preds, Y_train)
        end
        
        # Safely clip gradients
        clipped_grads = safe_gradient_clipping(grads)
        
        # Update model
        Flux.update!(state, model_cv, clipped_grads)
    end
    
    # Validate model
    Flux.reset!(model_cv)
    val_predictions = model_cv(X_val)
    val_mae = Flux.mae(val_predictions, Y_val)
    val_mse = Flux.mse(val_predictions, Y_val)
    
    return (model=model_cv, mae=val_mae, mse=val_mse)
end

function k_fold_cross_validation(data::Vector{Float64}, k::Int=5, seq_len::Int=64)
    # Prepare k-fold splits
    train_sets, val_sets = create_k_fold_splits(data, k, seq_len)
    
    # Store results for each fold
    fold_results = Vector{NamedTuple{(:model, :mae, :mse), Tuple{Chain, Float32, Float32}}}(undef, k)
    
    # Perform k-fold cross-validation
    for i in 1:k
        println("Training Fold $i")
        X_train, Y_train = train_sets[i]
        X_val, Y_val = val_sets[i]
        
        fold_results[i] = train_and_evaluate_model(X_train, Y_train, X_val, Y_val)
    end
    
    # Find the best model based on lowest MAE
    mae_values = [result.mae for result in fold_results]
    best_fold_index = argmin(mae_values)
    best_model = fold_results[best_fold_index].model
    
    # Compute average performance metrics
    avg_mae = mean(result.mae for result in fold_results)
    avg_mse = mean(result.mse for result in fold_results)
    
    println("\nCross-Validation Results:")
    println("Average MAE: ", avg_mae)
    println("Average MSE: ", avg_mse)
    println("Best Fold Index: ", best_fold_index)
    
    return best_model, fold_results, avg_mae, avg_mse
end

# Perform cross-validation
best_model, fold_results, avg_mae, avg_mse = k_fold_cross_validation(data.value)
rmse = sqrt(avg_mse) 

# Print detailed metrics
println("Mean Squared Error (MSE): ", avg_mse)
println("Mean Absolute Error (MAE): ", avg_mae)
println("Root Mean Squared Error (RMSE): ", rmse)


# Saving the best model from K fold approach.
BSON.@save "lstm_k_fold_model.bson" best_model


# Now we can Apply Conformal Prediction for Uncertainty Quantification.

# Set random seed for reproducibility
Random.seed!(123)

# --- Data Preparation with Train-Test Split ---
function prepare_train_test_split(df, test_size=0.2)
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
    
    return train_df, test_df, train_mean, train_std
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
function train_with_early_stopping(X_train, Y_train, X_val, Y_val; 
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
end

function train_conformal_predictor(model, X_calib, Y_calib, mean_val, std_val)
    # Reset LSTM state
    Flux.reset!(model)
    
    # Make predictions on calibration set
    predictions = model(X_calib)
    
    # Calculate absolute errors for calibration
    errors = abs.(predictions .- Y_calib)
    
    # Sort errors for quantile calculation
    sorted_errors = sort(errors)
    
    return ConformalPredictor(model, sorted_errors, mean_val, std_val)
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

# --- Evaluation Metrics for Conformal Prediction ---
function evaluate_conformal_predictions(Y_true, Y_pred, lower_bound, upper_bound)
    # Calculate coverage (percentage of true values within prediction intervals)
    coverage = mean((Y_true .>= lower_bound) .& (Y_true .<= upper_bound))
    
    # Calculate average interval width
    avg_width = mean(upper_bound .- lower_bound)
    
    # Calculate standard metrics
    mae = mean(abs.(Y_pred .- Y_true))
    mse = mean((Y_pred .- Y_true).^2)
    rmse = sqrt(mse)
    
    return (
        coverage = coverage,
        avg_width = avg_width,
        mae = mae,
        mse = mse,
        rmse = rmse
    )
end

# --- Main Execution ---
function run_conformal_prediction_pipeline()
    # Load data
    df = CSV.File("NAB/data/realTraffic/TravelTime_387.csv"; types=Dict(:value => Float64)) |> DataFrame
    
    # Split data into train, validation, and test sets
    train_df, test_df, train_mean, train_std = prepare_train_test_split(df, 0.2)
    
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
    best_model, train_losses, val_losses = train_with_early_stopping(X_train, Y_train, X_val, Y_val)
    
    # Prepare test data
    X_test, Y_test = create_sequences(test_df.normalized_value, seq_len)
    X_test, Y_test = Float32.(X_test), Float32.(Y_test)
    
    # Create validation+calibration set for conformal prediction
    X_calib, Y_calib = X_val, Y_val
    
    # Train conformal predictor
    conformal_predictor = train_conformal_predictor(best_model, X_calib, Y_calib, train_mean, train_std)
    
    # Make predictions with uncertainty
    point_preds, lower_bound, upper_bound = predict_with_intervals(conformal_predictor, X_test, 0.1)
    
    # Denormalize actual test values for comparison
    Y_test_original = (Y_test .* train_std) .+ train_mean
    
    # Evaluate conformal predictions
    metrics = evaluate_conformal_predictions(Y_test_original, point_preds, lower_bound, upper_bound)
    
    println("Conformal Prediction Results (90% confidence):")
    println("Coverage: $(round(metrics.coverage * 100, digits=2))%")
    println("Average Interval Width: $(round(metrics.avg_width, digits=2))")
    println("MAE: $(round(metrics.mae, digits=2))")
    println("RMSE: $(round(metrics.rmse, digits=2))")
    
    # Plot results
    p = plot(
        Y_test_original,
        label="Actual Values",
        linewidth=2,
        title="Traffic Prediction with Uncertainty\nCoverage: $(round(metrics.coverage * 100, digits=1))%, MAE: $(round(metrics.mae, digits=2))",
        xlabel="Time",
        ylabel="Travel Time",
        legend=:topright
    )
    
    plot!(p, point_preds, label="Predicted Values", linewidth=2, linestyle=:dash)
    
    # Plot prediction intervals as a ribbon
    plot!(
        p,
        collect(1:length(point_preds)),
        point_preds,
        ribbon=(point_preds .- lower_bound, upper_bound .- point_preds),
        fillalpha=0.3,
        label="90% Prediction Interval"
    )
    
    display(p)
    
    # Save results
    BSON.@save "traffic_conformal_model.bson" conformal_predictor metrics
    
    # Return for comparison
    return best_model, conformal_predictor, metrics
end

# Execute the pipeline
best_model, conformal_predictor, metrics = run_conformal_prediction_pipeline()

