export AbstractEstimator, AbstractEstimated
export DefaultEstimator, DefaultEstimated
export KalmanEstimator, LinearKalman, KalmanEstimated
export specifics, estimate
export measurement
export estimated
export modelproc, modelmeas
export initestim, initcontrol
export scanestimator
    
abstract type AbstractEstimator end
struct DefaultEstimator <: AbstractEstimator end
abstract type KalmanEstimator <: AbstractEstimator end
abstract type LinearKalman <: KalmanEstimator end

abstract type AbstractEstimated{T} end
struct DefaultEstimated{T} <: AbstractEstimated{T}
    x::AbstractVector{T}
end
estimated(::DefaultEstimator) = DefaultEstimated
initestim(e::DefaultEstimator, v::AbstractVector{T}=[]) where T = estimated(e)(v)
initcontrol(::DefaultEstimator) = []

mutable struct KalmanEstimated{T} <: AbstractEstimated{T}
    P::AbstractMatrix{T}
    x::AbstractVector{T}
end
estimated(::KalmanEstimator) = KalmanEstimated

measurement(::AbstractEstimator, y) = Vector(y)
modelproc(::AbstractEstimator)::Model = missing
modelmeas(::AbstractEstimator)::Model = missing

initestim(e::LinearKalman, v::AbstractVector{T}) where T = estimated(e)(T.(I(length(v)) * 1e6), v)
initestim(e::LinearKalman, v::AbstractVector{Particles{T}}) where T = estimated(e)(T.(I(length(v)) * 1e6) .+ 0 .* Particles.(eltype(v).parameters[2]), v)
initestim(e::LinearKalman) = initestim(e, zeros(xsize(modelproc(e))))
initcontrol(e::KalmanEstimator) = zeros(usize(modelproc(e)))

"Update function of Kalman filters"
function estimate(e::KalmanEstimator,
                  x::KalmanEstimated{T},
                  y::AbstractVector{T},
                  u::AbstractVector{T} =
                      zeros(usize(modelproc(e)))) where T
    P = x.P
    x = x.x
    V = typeof(x)
    process = modelproc(e)
    measure = modelmeas(e)
    x = V(process(;x, u, y))
    P = process(P, x, u, y)
    S = measure(P, x, u, y)
    C = A(measure; x, u, y) # measurement model's A is actually C
    z = measure(;x, u, y)
    K = P * C' * pinv(S)
    # from above KSK' == PC'K'
    # == (KCP')' =(P symm)= (KCP)' =(innovaton to P is sym)= KCP
    # instead of
    # P -= K * C * P
    # we write this to retain symmetry and positive definitness
    P -= K * S * K'
    x += K * (measurement(e, y) - z)
    KalmanEstimated(P, V(x))
end

function scanestimator(
    e::KalmanEstimator;
    y,
    u=Iterators.repeated(initcontrol(e)),
    init=initestim(e))
    out = zip(y, u) |> Scan(init) do x, (y, u)
        estimate(e, x, y, u)
    end
    x = [e.x for e in out]
    P = [e.P for e in out]
    (;P, x)
end

function estimate(::AbstractEstimator, 
                  ::AbstractEstimated{T},
                  y::AbstractVector{T},
                  ::AbstractVector{T}) where T
    DefaultEstimated(y)
end
