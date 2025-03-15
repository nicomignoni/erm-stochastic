using Dates, Convex, HiGHS
import Convex: MOI

include("model.jl")

@kwdef struct CvxDecision{H}
    a::Variable = Variable(H, Positive())
    b::Variable = Variable(H, Positive())
    q::Variable = Variable(H)
    u::Variable = Variable(H)
    z::Variable = Variable(H, BinVar)
end

function upper_bounds(x::CvxDecision, agent::Agent)
    return [abs(x.q) <= agent.q̄, x.a <= agent.φ * agent.b̄, x.b <= (1 - agent.φ) * agent.b̄]
end

function z_definition(x::CvxDecision, agent::Agent, ϵ::Real)
    return [x.q >= agent.q̄ .* (x.z - 1), x.q <= (agent.q̄ + ϵ) .* x.z - ϵ]
end

function u_definition(x::CvxDecision, agent::Agent)
    return [
        x.u <= agent.q̄ * x.z, x.u >= -agent.q̄ * x.z,
        x.u <= x.q + agent.q̄ * (1 - x.z), x.u >= x.q - agent.q̄ * (1 - x.z)
    ]
end

function a_definition(x::CvxDecision, agent::Agent, Δ::Real, a₀::Real)
    return [x.a == (agent.α - agent.Γ) * vcat(a₀, x.a[1:end-1]) + Δ * (x.q / agent.η⁻ + x.u * (agent.η⁺ - 1 / agent.η⁻)) + agent.Γ * x.b]
end

function b_definition(x::CvxDecision, agent::Agent, b₀::Real)
    return x.b == (agent.α - agent.Γ) * vcat(b₀, x.b[1:end-1]) + agent.Γ * x.a
end

function degradation(x::CvxDecision, agent::Agent)
    return abs.(x.q) / ((1 + 1 / agent.η⁺) * agent.L * agent.b̄)
end

function rebalance_cost(x::CvxDecision, ξ::Uncertainty, retailer::Retailer)
    energy_bought = ξ.δ .- ξ.γ .+ x.q
    return max.(energy_bought .* retailer.ρ⁺, energy_bought .* retailer.ρ⁻)
end

function mpc(
    K::Int, agent::Agent, retailer::Retailer, ξ::Uncertainty,
    a₀::Real, b₀::Real, Δ::Real, ϵ::Real, solver
)
    x = CvxDecision{K}()
    constraints = [
        upper_bounds(x, agent);
        z_definition(x, agent, ϵ);
        u_definition(x, agent);
        a_definition(x, agent, Δ, a₀);
        b_definition(x, agent, b₀)
    ]
    objective = sum(agent.C * degradation(x, agent) .+ rebalance_cost(x, ξ, retailer))
    problem = minimize(objective, constraints)
    solve!(problem, solver; silent=true)
    return Decision(x.a.value, x.b.value, x.q.value, x.u.value, x.z.value)
end
