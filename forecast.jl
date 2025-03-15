using StateSpaceModels, CSV, DataFrames, ProgressMeter, JLD

include("model.jl")
include("settings.jl")

# Read data and convert in kW and transform power in energy (kWh)
data = CSV.read(DATA_PATH, DataFrame; delim=",", dateformat="dd/mm/yyyy HH:MM")
transform!(data, [:generation, :demand] .=> ByRow(x -> 1e-3Δ * x) .=> [:generation, :demand])

forecast = Vector{Uncertainty}(undef, K + 1)
@showprogress for k in t₀:t₀+K
    # Historical data
    generation_history = data.generation[k-l:k-1]
    demand_history = data.demand[k-l:k-1]

    # Instantiate forecast models
    generation_model = UnobservedComponents(generation_history)
    demand_model = UnobservedComponents(demand_history)

    fit!(generation_model, save_hyperparameter_distribution=false)
    fit!(demand_model, save_hyperparameter_distribution=false)

    forecast[k-t₀+1] = Uncertainty(
        max.(0, dropdims(simulate_scenarios(generation_model, K + 1, n), dims=2)),
        max.(0, dropdims(simulate_scenarios(demand_model, K + 1, n), dims=2))
    )
end

save("result/forecast.jld", "forecast", forecast)





