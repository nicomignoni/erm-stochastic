using Dates, LinearAlgebra, Distributions, CSV, DataFrames, JLD, CairoMakie, Makie.Colors

include("settings.jl")
include("mpc.jl")

# Read ground truth and convert in kW and transform power in energy (kWh)
truth = CSV.read(DATA_PATH, DataFrame; delim=",", dateformat="dd/mm/yyyy HH:MM")
transform!(truth, [:generation, :demand] .=> ByRow(x -> 1e-3Δ * x) .=> [:generation, :demand])

# Read results
result = load("result/result.jld", "result")
Kₐ = length(result)
Kₚ = Kₐ

variables = [:a, :b, :q]
uncertainty = [:γ, :δ]

decision_fig = Figure(size=(3.5INCH, 3INCH), figure_padding=1, fonts=(; regular="Computer Modern"))
uncertain_fig = Figure(size=(3.5INCH, 2INCH), figure_padding=1, fonts=(; regular="Computer Modern"))

axis_kwargs = Dict(
    :xticklabelsize => 8,
    :yticklabelsize => 8,
    :xticklabelsvisible => false,
    :xticks => 1:14:K,
    :xlabelsize => 10,
    :ylabelsize => 10,
)

axs = Dict(
    "x" => Dict(
        v => Axis(decision_fig[i, 1];
            ylabel=v == :q ? L"\text{Power [kW]}" : L"\text{Energy [kWh]}",
            axis_kwargs...)
        for (i, v) in enumerate(variables)
    ),
    "ξ" => Dict(
        v => Axis(uncertain_fig[i, 1];
            ylabel=L"\text{Power [kW]}",
            axis_kwargs...)
        for (i, v) in enumerate(uncertainty)
    )
)

actionable_decision = Dict(v => Vector{Float16}(undef, Kₐ) for v in variables)
for (k, state) in enumerate(result)
    # Decision
    for (i, v) in enumerate(variables)
        var = getfield(state.x, v)
        lines!(axs["x"][v], k:min(Kₚ + k - 1, Kₐ), var[1:min(Kₚ, Kₐ - k + 1), 1], color=8, colorrange=(1, 10), colormap=:grays, linewidth=0.5)
        actionable_decision[v][k] = var[1, 1]
    end

    # Uncertainty
    for (i, u) in enumerate(uncertainty)
        unc = getfield(state.ξ, u)
        scatter!(axs["ξ"][u], Point2f.(k, unc[1, :]), markersize=1, color=8, colorrange=(1, 10), colormap=:grays)
    end
end

line_color = colorant"#fe8100"
# Plot actionable decision
for (i, v) in enumerate(variables)
    lines!(axs["x"][v], 1:Kₐ, actionable_decision[v])
end

# Plot ground truth
lines!(axs["ξ"][:γ], truth.generation[t₀:t₀+Kₐ])
lines!(axs["ξ"][:δ], truth.demand[t₀:t₀+Kₐ])

# Plot xlabel
for (i, v) in [("x", :q), ("ξ", :δ)]
    axs[i][v].xlabel = L"\text{Time [h]}"
    axs[i][v].xtickformat = (i -> Dates.format.(truth.time[t₀.+Int.(i)], "HH:MM"))
    axs[i][v].xticklabelsvisible = true
end

save(FIG_DIR * "generation_demand.pdf", uncertain_fig)
save(FIG_DIR * "result.pdf", decision_fig)
