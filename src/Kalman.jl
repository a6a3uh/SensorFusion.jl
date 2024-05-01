export Kalman, LinearKalman
export specifics, update, estimate_batch
export measurement
    
abstract type Kalman end
struct LinearKalman <: Kalman end

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
    ỹ::AbstractVector,
    u=0)
    x = process(x, u, ỹ)
    y = measure(x, u, ỹ)
    P = process(P, x, u, ỹ)
    S = measure(P, x, u, ỹ)
    W = P * A(measure, x, u, ỹ)' # measurement model's A is actually C
    (;x, y, P, S, W)
end

"Update function of Kalman filters"
update(process, measure, method) = (
    P::AbstractMatrix,
    x::AbstractVector,
    ỹ::AbstractVector,
    u = zeros(usize(process))) -> let
    x, y, P, S, W = specifics(method, process, measure, P, x, ỹ, u)
    F = W * pinv(S)
    P = P - F * S * F'
    P, x + F * (ỹ - y) # broadcasting for scalar measurements
end;

function estimate_batch(process::Model,
                  measure::Model,
                  method::Kalman,
                  P₀, x₀, ys,
                  us=Iterators.repeated(0))
    P, x = zip(ys, us) |> Scan(
	(P₀, x₀)) do (P, x), (y, u) 
	    update(process, measure, method)(P, x, y, u)
	end |> xs -> zip(xs...) |> collect
end
