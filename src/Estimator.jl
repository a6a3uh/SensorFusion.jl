export AbstractEstimator, AbstractEstimated
export DefaultEstimator, DefaultEstimated

abstract type AbstractEstimator end

struct DefaultEstimator <: AbstractEstimator end

abstract type AbstractEstimated{T} end

struct DefaultEstimated{T} <: AbstractEstimated{T}
    x::AbstractVector{T}
end
