include("./Estimator.jl")

using LinearAlgebra
using Models

export KalmanEstimator, LinearKalman, KalmanEstimated
# export modelproc, modelmeas
export measurement
export process_update, measurement_update
    
abstract type KalmanEstimator <: AbstractEstimator end
abstract type LinearKalman <: KalmanEstimator end

mutable struct KalmanEstimated{T} <: AbstractEstimated{T}
    P::AbstractMatrix{T}
    x::AbstractVector{T}
end

function KalmanEstimated(v::AbstractVector{T}) where T
    KalmanEstimated(T.(I(length(v)) * 1e6), v)
end

# modelproc(::AbstractEstimator)::Model = missing
# modelmeas(::AbstractEstimator)::Model = missing
# measurement(::Model, y) = Vector(y)
measurement(::Model, y) = y

function measurement_update(m::Model{T};
                            x::KalmanEstimated{T},
                            y::AbstractVector{T},
                            u::AbstractVector{T}=T[0]) where T
    P = x.P
    x = x.x
    V = typeof(x)
    
    S = m(P; x, u, y)
    C = A(m; x, u, y) # measurement model's A is actually C
    K = P * C' * pinv(S)
    P -= K * S * K'
    δy = measurement(m, y) - m(;x, u, y)
    δx = K * δy
    x = x + δx
    KalmanEstimated((P + P') / 2, V(x))
end

function process_update(
    m::Model{T};
    x::KalmanEstimated{T},
    y::AbstractVector{T}=T[],
    u::AbstractVector{T}=T[0],
    δt=0.) where T
    P = x.P
    x = x.x
    V = typeof(x)
    x = V(m(;x, u, y, δt))
    P = m(P; x, u, y, δt)
    KalmanEstimated((P + P') / 2, x)
end
