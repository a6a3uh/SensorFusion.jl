include("./Estimator.jl")

using LinearAlgebra
using Models

export KalmanEstimator, LinearKalman, KalmanEstimated
export measurement
export process_update, propagate, update
    
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

function update(m::Model{T};
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

function propagate(
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

function update(
    ms::Vector{<:Model{T}};
    x::KalmanEstimated{T},
    ys,
    u::AbstractVector{T}=T[0]) where T

    P = x.P
    x = x.x
    V = typeof(x)

    x, H, HtSi = reduce(zip(ms, ys), init=(x, [], [])) do (x, H, HtSi), (m, y)
        δy = measurement(m, y) - m(;x, u, y)
        s = m(P; x, u, y)
        h = A(m; x, u, y)
        htsi = h'pinv(s)
        k = P * htsi
        δx = k * δy
        x = V(x + δx)
        H = length(H) == 0 ? h : vcat(H, h)
        HtSi = length(HtSi) == 0 ? htsi : hcat(HtSi, htsi)
        (x, H, HtSi)
    end
    K = P * HtSi
    
    P -= K * H * P
    
    KalmanEstimated((P + P') / 2, V(x))
end
