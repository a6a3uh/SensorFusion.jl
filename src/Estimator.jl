include("./Models.jl")

export AbstractEstimator, AbstractEstimated
export DefaultEstimator, DefaultEstimated
export estimate

abstract type AbstractEstimator end

struct DefaultEstimator <: AbstractEstimator end

abstract type AbstractEstimated{T} end

struct DefaultEstimated{T} <: AbstractEstimated{T}
    x::AbstractVector{T}
end

function estimate(::AbstractEstimator, 
                  ::AbstractEstimated{T},
                  y::AbstractVector{T},
                  ::AbstractVector{T}) where T
    DefaultEstimated(y)
end
