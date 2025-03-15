using Dates

const CONTRIBUTO_AL_CONSUMO = 0.041  # €/kWh
const ENEL_FLEXBOX_PRICES = [0.15832, 0.15161, 0.12854] # €/kWh
const RITIRO_DEDICATO_PRICE = 0.01 # €/kWh

@kwdef struct Agent
    α::Real # leakage
    η⁺::Real # inflow efficiency
    η⁻::Real # outflow efficiency
    q̄::Real # max storage exchange
    b̄::Real # max storage
    φ::Real # available storage fraction
    Γ::Real # transfer rate
    C::Real # end of life cost
    L::Real # rate life cycle
end

@kwdef struct Retailer
    ρ⁺::Vector{Real} # price retailer sells to the prosumer
    ρ⁻::Vector{Real} # price prosumer sells to the retailer
end

struct Decision
    a::Matrix{Float16}
    b::Matrix{Float16}
    q::Matrix{Float16}
    u::Matrix{Float16}
    z::Matrix{Float16}
end

struct Uncertainty
    γ::Matrix{Float16} # generation
    δ::Matrix{Float16} # demand
end

struct State
    t::DateTime
    x::Decision
    ξ::Uncertainty
end

function fascia(date::DateTime)
    """
    Time period subdivision for the Italian energy market.
    See: https://www.enel.it/it/offerte/luce/offerte/enel-flex-box (Accessed 08-02-2025)
    """
    weekday = Dates.dayofweek(date)
    hour = Dates.hour(date)

    if weekday in 1:5 && hour in 8:20
        return 1
    elseif (weekday == 6 && hour in 7:23) || (weekday in 1:5 && (hour in 7:9 || hour in 19:23))
        return 2
    else
        return 3
    end
end

"""
Retailer pricing scheme based on three time period, based on Enel Flex Box plan.
See: https://www.enel.it/it/offerte/luce/offerte/enel-flex-box (Accessed 08-02-2025)
"""
enel_flex_box(date::DateTime) = CONTRIBUTO_AL_CONSUMO + ENEL_FLEXBOX_PRICES[fascia(date)]

"""
Retailer pricing scheme for buying from prosumers.
See: https://www.gse.it/servizi-per-te/fotovoltaico/ritiro-dedicato
"""
ritiro_dedicato(date::DateTime) = RITIRO_DEDICATO_PRICE
