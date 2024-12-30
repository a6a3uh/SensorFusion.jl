include("./Estimator.jl")

using LinearAlgebra
using Models

export KalmanEstimator, LinearKalman, KalmanEstimated
export measurement
export process_update, propagate, update, update_batch, update_seq
    
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

function propagate(
    m::Model{T};
    e::KalmanEstimated{T},
    y::AbstractVector{T}=T[],
    u::AbstractVector{T}=T[0],
    δt=0.) where T
    V = typeof(e.x)
    x = V(m(;x=e.x, u, y, δt))
    P = m(e.P; x=e.x, u, y, δt)
    KalmanEstimated((P + P') / 2, x)
end

function update(m::Model{T};
                e::KalmanEstimated{T},
                y::AbstractVector{T},
                u::AbstractVector{T}=T[0]) where T
    
    S = m(e.P; x=e.x, u, y)
    H = A(m; x=e.x, u, y)
    K = e.P * H' * pinv(S)
    P = e.P - K * S * K'
    δy = measurement(m, y) - m(;x=e.x, u, y)
    δx = K * δy
    x = e.x + δx
    KalmanEstimated((P + P') / 2, typeof(e.x)(x))
end

function update(
    ms::Vector{<:Model{T}};
    e::KalmanEstimated{T},
    ys,
    u::AbstractVector{T}=T[0],
    batch=true) where T
    batch ? update_batch(ms; e, ys, u) : update_seq(ms; e, ys, u)
end

function update_seq(
    ms::Vector{<:Model{T}};
    e::KalmanEstimated{T},
    ys,
    u::AbstractVector{T}=T[0]) where T
    reduce((e, (m, y)) -> update(m; e, y, u), zip(ms, ys), init=e)
end

function update_batch(
    ms::Vector{<:Model{T}};
    e::KalmanEstimated{T},
    ys,
    u::AbstractVector{T}=T[0]) where T

    x, H, HtSi = reduce(zip(ms, ys), init=(e.x, [], [])) do (x, H, HtSi), (m, y)
        s = m(e.P; x, u, y)
        h = A(m; x, u, y)
        htsi = h'pinv(s)
        k = e.P * htsi
        δy = measurement(m, y) - m(;x, u, y)
        δx = k * δy
        x = typeof(e.x)(x + δx)
        H = length(H) == 0 ? h : vcat(H, h)
        HtSi = length(HtSi) == 0 ? htsi : hcat(HtSi, htsi)
        (x, H, HtSi)
    end
    K = e.P * HtSi
    
    P = e.P - K * H * e.P
    
    KalmanEstimated((P + P') / 2, x)
end
