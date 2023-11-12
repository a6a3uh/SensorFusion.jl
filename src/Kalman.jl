abstract type Kalman end
struct LinearKalman <: Kalman end

"Part of Kalman filter specific for
model based (KF, EKF) and not sampling based like (UKF)
cov propagation"
function specifics(
        method::LinearKalman,
        process::Model,
        measure::Model,
        P::AbstractMatrix,
        x::AbstractVector,
        u=0)
    x = process(x, u)
    y = measure(x, u)
    P = process(P, x)
    S = measure(P, x)
    W = P * A(measure, x)' # measurement model's A is actually C
    (;x, y, P, S, W)
end

"Update function of Kalman filters"
update(process, measure, method) = (
    P::AbstractMatrix,
    x::AbstractVector,
    ỹ::AbstractVector,
    u = 0) -> let
    x, y, P, S, W = specifics(method, process, measure, P, x, u)
    F = W * pinv(S)
    P = P - F * S * F'
    P, x + F * (ỹ - y)
end;

function estimate(process::Model,
                  measure::Model,
                  method::Kalman,
                  P₀, x₀, ys,
                  us=Iterators.repeated(0))
    P, x = zip(ys, us) |> Scan(
	(P₀, x₀)) do (P, x), (y, u) 
	    update(process, measure, method)(P, x, y, u)
	end |> xs -> zip(xs...) |> collect
end
