include("./Estimator.jl")

using LinearAlgebra
using Models

export KalmanEstimator, LinearKalman, KalmanEstimated
export measurement
export process_update, measurement_update, measurements_update
    
abstract type KalmanEstimator <: AbstractEstimator end
abstract type LinearKalman <: KalmanEstimator end

mutable struct KalmanEstimated{T} <: AbstractEstimated{T}
    P::AbstractMatrix{T}
    x::AbstractVector{T}
end

function KalmanEstimated(v::AbstractVector{T}) where T
    KalmanEstimated(T.(I(length(v)) * 1e6), v)
end

measurement(::Model, y) = y

function measurements_update(
    ms::Vector{Model{T}};
    x::KalmanEstimated{T},
    ys,
    u::AbstractVector{T}=T[0]) where T

    P = x.P
    x = x.x
    V = typeof(x)
    
    H = reduce(vcat, A(m; x, u, y) for (m, y) in zip(ms, ys))
    K = P * reduce(hcat, A(m; x, u, y)' * pinv(m(P; x, u, y)) for (m, y) in zip(ms, ys))

    P -= K * H * P

    δxs = map(zip(ms, ys)) do (m, y)
        δy = measurement(m, y) - m(;x, u, y)
        s = m(P; x, u, y)
        h = A(m; x, u, y)
        k = P * h' * pinv(s)
        δx = k * δy
    end
    
    x = x + sum(δxs) 
    KalmanEstimated((P + P') / 2, V(x))
end

function measurement_update(m::Model{T};
                            x::KalmanEstimated{T},
                            y::AbstractVector{T},
                            u::AbstractVector{T}=T[0]) where T
    P = x.P
    x = x.x
    V = typeof(x)
    
    S = m(P; x, u, y)
    H = A(m; x, u, y)
    K = P * H' * pinv(S)
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
