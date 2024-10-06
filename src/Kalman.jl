include("./Estimator.jl")

using LinearAlgebra

export KalmanEstimator, LinearKalman, KalmanEstimated
export modelproc, modelmeas
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

modelproc(::AbstractEstimator)::Model = missing
modelmeas(::AbstractEstimator)::Model = missing
measurement(::Model, y) = Vector(y)

function measurement_update(m::Model,
                            x::KalmanEstimated{T},
                            y::AbstractVector{T},
                            u::AbstractVector{T}=zeros(T, usize(m))) where T
    P = x.P
    x = x.x
    V = typeof(x)
    
    S = m(P, x, u, y)
    C = A(m; x, u, y) # measurement model's A is actually C
    z = m(;x, u, y)
    K = P * C' * pinv(S)
    # K = P * C' / S # not works
    # K = (pinv(S)' * C * P')' == (pinv(S) * C * P)' == (S \ C * P)' 
    # K = (S \ C * P)' # not works
    # K = P * (S \ C)' # not works either
    #
    # from above KSK' == PC'K'
    # == (KCP')' =(P symm)= (KCP)' =(innovaton to P is sym)= KCP
    # instead of
    # P -= K * C * P # or equivalently P = (I - KC)P
    # we write this to retain symmetry and positive definitness
    P -= K * S * K'
    P = (P + P') / 2 # one more time to ensure symmetry of P
    x += K * (measurement(m, y) - z)
    KalmanEstimated(P, V(x))
end

function process_update(m::Model,
                        x::KalmanEstimated{T},
                        y::AbstractVector{T}=zeros(T, ysize(m)),
                        u::AbstractVector{T}=zeros(T, usize(m))) where T
    P = x.P
    x = x.x
    V = typeof(x)
    x = V(m(;x, u, y))
    P = m(P, x, u, y)
    KalmanEstimated(P, x)
end
    
"Update function of Kalman filters"
function estimate(e::KalmanEstimator,
                  x::KalmanEstimated{T},
                  y::AbstractVector{T},
                  u::AbstractVector{T} =
                      zeros(usize(modelproc(e)))) where T

    x = process_update(modelproc(e), x, y, u)
    return measurement_update(modelmeas(e), x, y, u)
end
