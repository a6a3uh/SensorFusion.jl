export AbstractEstimator, AbstractEstimated
export DefaultEstimator, DefaultEstimated
export KalmanEstimator, LinearKalman
export specifics, update, estimate_batch
export measurement
export estimated
    
abstract type AbstractEstimator end
struct DefaultEstimator <: AbstractEstimator end
abstract type KalmanEstimator <: AbstractEstimator end
abstract type LinearKalman <: KalmanEstimator end

abstract type AbstractEstimated{T} end
struct DefaultEstimated{T} <: AbstractEstimated{T}
    x::AbstractVector{T}
end
DefaultEstimated(T::Type) = DefaultEstimated(T[])

mutable struct KalmanEstimated{T} <: AbstractEstimated{T}
    P::AbstractMatrix{T}
    x::AbstractVector{T}
end

estimated(::DefaultEstimator) = DefaultEstimated
estimated(::KalmanEstimator) = KalmanEstimated

# abstract type Kalman end
# struct LinearKalman <: Kalman end

measurement(::Model, y) = Vector(y)

"Part of Kalman filter specific for
model based (KF, EKF) and not sampling based like (UKF)
cov propagation"
function specifics(
    ::LinearKalman,
    process::Model,
    measure::Model,
    P::AbstractMatrix,
    x::AbstractVector,
    y::AbstractVector,
    u=0)
    x = process(;x, u, y)
    P = process(P, x, u, y)
    S = measure(P, x, u, y)
    W = P * A(measure; x, u, y)' # measurement model's A is actually C
    y = measure(;x, u, y)
    (;x, y, P, S, W)
end

"Update function of Kalman filters"
update(process::Model, measure::Model, method::KalmanEstimator) = (
    x::KalmanEstimated,
    # P::AbstractMatrix,
    # x::AbstractVector,
    ỹ::AbstractVector,
    u = zeros(usize(process))) -> let
    x, y, P, S, W = specifics(method, process, measure, x.P, x.x, ỹ, u)
    F = W * pinv(S)
    P -= F * S * F'
    x += F * (measurement(measure, ỹ) - y) # broadcasting for scalar measurements
    KalmanEstimated(P, x)
end;

function estimate_batch(process::Model,
                  measure::Model,
                  method::KalmanEstimator,
                  P₀, x₀, ys,
                  us=Iterators.repeated(0))
    P, x = zip(ys, us) |> Scan(
	(P₀, x₀)) do (P, x), (y, u) 
	    update(process, measure, method)(P, x, y, u)
	end |> xs -> zip(xs...) |> collect
end
