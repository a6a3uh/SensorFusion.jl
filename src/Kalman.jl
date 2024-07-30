export AbstractEstimator, AbstractEstimated
export DefaultEstimator, DefaultEstimated
export KalmanEstimator, LinearKalman
export specifics, update, estimate_batch
export measurement
export estimated
export modelproc, modelmeas
export initestim
export scanestimator
    
abstract type AbstractEstimator end
struct DefaultEstimator <: AbstractEstimator end
abstract type KalmanEstimator <: AbstractEstimator end
abstract type LinearKalman <: KalmanEstimator end

abstract type AbstractEstimated{T} end
struct DefaultEstimated{T} <: AbstractEstimated{T}
    x::AbstractVector{T}
end
DefaultEstimated(T::Type) = DefaultEstimated(T[])
estimated(::DefaultEstimator) = DefaultEstimated

mutable struct KalmanEstimated{T} <: AbstractEstimated{T}
    P::AbstractMatrix{T}
    x::AbstractVector{T}
end
estimated(::KalmanEstimator) = KalmanEstimated

measurement(::Model, y) = Vector(y)
modelproc(::AbstractEstimator)::Model = missing
modelmeas(::AbstractEstimator)::Model = missing
initestim(e::LinearKalman) = KalmanEstimated(
    I(xsize(modelproc(e))), zeros(xsize(modelproc(e))))

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
update(e::KalmanEstimator) = (
    x::KalmanEstimated,
    ỹ::AbstractVector,
    u = zeros(usize(modelproc(e)))) -> let
        x, y, P, S, W = specifics(
            e,
            modelproc(e),
            modelmeas(e), x.P, x.x, ỹ, u)
    F = W * pinv(S)
    P -= F * S * F'
    x += F * (measurement(modelmeas(e), ỹ) - y) # broadcasting for scalar measurements
    KalmanEstimated(P, x)
end;

function scanestimator(e::KalmanEstimator; y, u=Iterators.repeated(0.0))
    out = zip(y, u) |> Scan(initestim(e)) do x, (y, u)
        update(e)(x, y, u)
    end
    x = [e.x for e in out]
    P = [e.P for e in out]
    (;P, x)
end

