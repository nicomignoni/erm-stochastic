using Dates, LinearAlgebra, Random, Distributions, DataFrames, CSV, StateSpaceModels, ProgressMeter, JLD
import StateSpaceModels: Optim

include("settings.jl")
include("model.jl")
include("mpc.jl")

Random.seed!(2024)

MPC_SOLVER = MOI.OptimizerWithAttributes(HiGHS.Optimizer,
    "mip_rel_gap" => 0.03,
    "threads" => 5,
    MOI.Silent() => true
)
FORECAST_SOLVER = Optimizer(Optim.GradientDescent())

# Initial storage
a₀ = 0
b₀ = 0

# Read data and convert in kW and transform power in energy (kWh)
data = CSV.read(DATA_PATH, DataFrame; delim=",", dateformat="dd/mm/yyyy HH:MM")
transform!(data, [:generation, :demand] .=> ByRow(x -> 1e-3Δ * x) .=> [:generation, :demand])

agent = Agent(α=0.99, η⁺=0.99, η⁻=0.98, q̄=2, b̄=5, φ=0.5, Γ=0.1, C=300, L=3000)

# Main loop
result = Vector{State}(undef, K + 1)
@showprogress for k in t₀:t₀+K
    # Get retailer prices
    time_horizon = data.time[k:k+K]
    retailer = Retailer(ρ⁺=enel_flex_box.(time_horizon), ρ⁻=ritiro_dedicato.(time_horizon))

    # Historical data
    generation_history = data.generation[k-l:k-1]
    demand_history = data.demand[k-l:k-1]

    # Instantiate forecast models
    generation_model = UnobservedComponents(generation_history)
    demand_model = UnobservedComponents(demand_history)

    fit!(generation_model, save_hyperparameter_distribution=false)
    fit!(demand_model, save_hyperparameter_distribution=false)

    ξ = Uncertainty(
        max.(0, dropdims(simulate_scenarios(generation_model, K + 1, n), dims=2)),
        max.(0, dropdims(simulate_scenarios(demand_model, K + 1, n), dims=2))
    )

    # MPC step
    x = mpc(K + 1, agent, retailer, ξ, a₀, b₀, Δ, ϵ, MPC_SOLVER)

    # Update history
    t = data.time[k]
    result[k-t₀+1] = State(t, x, ξ)

    # Update initial storage values
    global a₀, b₀ = x.a[1, 1], x.b[1, 1]
end

# Save result
save(RESULT_DIR * "result.jld", "result", result)






